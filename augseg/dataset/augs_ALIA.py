import numpy as np
import random
import torch
import scipy.stats as stats
import scipy.ndimage as ndi

# Optional: use SLIC if available
try:
    from skimage.segmentation import slic
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# # # # # # # # # # # # # # # # # # # # #
# # 0 random box
# # # # # # # # # # # # # # # # # # # # #
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # #
# # 1 helper: tensor -> image for slic
# # # # # # # # # # # # # # # # # # # # #
def _tensor_to_hwc_uint8(img_tensor):
    """
    img_tensor: [3, H, W], normalized image
    returns : HWC uint8 image
    """
    img = img_tensor.detach().cpu().float().clone()

    # assume ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)

    img = img * std + mean
    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return img


# # # # # # # # # # # # # # # # # # # # #
# # 2 helper: shape-aware mask from low-confidence
# # # # # # # # # # # # # # # # # # # # #
def _build_shape_aware_region(unlabeled_img_i, unlabeled_logits_i, image_confidence):
    """
    unlabeled_img_i   : [3, H, W]
    unlabeled_logits_i: [H, W]   (pixel-wise confidence/logit map already used in your code)
    image_confidence  : float    (sample-level confidence from train_semi.py)

    returns:
        selected_mask: numpy bool [H, W]
    """
    H, W = unlabeled_logits_i.shape

    conf_map = unlabeled_logits_i.detach().cpu().numpy()

    # low-confidence pixels
    # sample-level confidence is used adaptively; clamp for safety
    thr = float(np.clip(image_confidence, 0.05, 0.95))
    low_conf_mask = conf_map < thr

    # if too few pixels, fallback to lowest 20%
    if low_conf_mask.sum() < max(64, int(0.01 * H * W)):
        adaptive_thr = np.percentile(conf_map, 20)
        low_conf_mask = conf_map <= adaptive_thr

    # -------- Option A: SLIC superpixel mask --------
    if _HAS_SKIMAGE:
        try:
            img_hwc = _tensor_to_hwc_uint8(unlabeled_img_i)
            segments = slic(
                img_hwc,
                n_segments=120,
                compactness=10.0,
                sigma=1.0,
                start_label=0,
            )

            unique_ids = np.unique(segments)
            scored = []

            for sid in unique_ids:
                sp_mask = (segments == sid)
                area = sp_mask.sum()
                if area == 0:
                    continue

                overlap = np.logical_and(sp_mask, low_conf_mask).sum()
                if overlap == 0:
                    continue

                overlap_ratio = overlap / (area + 1e-6)
                scored.append((sid, overlap_ratio, overlap, area))

            if len(scored) > 0:
                scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

                # target mixed area: 25% ~ 45%
                target_ratio = np.random.uniform(0.25, 0.45)
                target_pixels = int(target_ratio * H * W)

                selected = np.zeros((H, W), dtype=np.bool_)
                current_pixels = 0

                for sid, _, _, _ in scored:
                    selected |= (segments == sid)
                    current_pixels = int(selected.sum())
                    if current_pixels >= target_pixels:
                        break

                if selected.sum() > 0:
                    return selected
        except Exception:
            pass

    # -------- Option B: fallback using connected low-confidence regions --------
    # This is still shape-aware because it follows low-confidence region geometry,
    # not a random rectangle.
    labeled_cc, num_cc = ndi.label(low_conf_mask.astype(np.uint8))
    if num_cc > 0:
        component_ids = list(range(1, num_cc + 1))
        component_sizes = [(labeled_cc == cid).sum() for cid in component_ids]
        sorted_ids = [cid for _, cid in sorted(zip(component_sizes, component_ids), reverse=True)]

        selected = np.zeros((H, W), dtype=np.bool_)
        target_ratio = np.random.uniform(0.20, 0.40)
        target_pixels = int(target_ratio * H * W)

        for cid in sorted_ids:
            selected |= (labeled_cc == cid)
            if selected.sum() >= target_pixels:
                break

        if selected.sum() > 0:
            return selected

    # -------- Final fallback: rectangle --------
    lam = np.random.beta(8, 2)
    bbx1, bby1, bbx2, bby2 = rand_bbox((1, H, W), lam=lam)
    selected = np.zeros((H, W), dtype=np.bool_)
    selected[bbx1[0]:bbx2[0], bby1[0]:bby2[0]] = True
    return selected


# # # # # # # # # # # # # # # # # # # # #
# # 3 cutmix label-adaptive (shape-aware version)
# # # # # # # # # # # # # # # # # # # # #
def cut_mix_label_adaptive(unlabeled_image, unlabeled_mask, unlabeled_logits,
                           labeled_image, labeled_mask, lst_confidences):
    assert len(lst_confidences) == len(unlabeled_image), "Ensure the confidence is properly obtained"
    assert labeled_image.shape == unlabeled_image.shape, "Ensure shape match between lb and unlb"

    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    labeled_logits = torch.ones_like(unlabeled_logits)

    # 1) random index for pairing
    u_rand_index = torch.randperm(unlabeled_image.size(0))[:unlabeled_image.size(0)]

    # 2) first stage: inject labeled information into low-confidence regions
    for i in range(mix_unlabeled_image.shape[0]):
        if np.random.random() > lst_confidences[i]:
            paste_mask = _build_shape_aware_region(
                unlabeled_img_i=unlabeled_image[i],
                unlabeled_logits_i=unlabeled_logits[i],
                image_confidence=lst_confidences[i]
            )

            paste_mask_t = torch.from_numpy(paste_mask).to(unlabeled_image.device)

            src_idx = u_rand_index[i]

            # image
            mix_unlabeled_image[i, :, paste_mask_t] = labeled_image[src_idx, :, paste_mask_t]

            # target
            mix_unlabeled_target[i, paste_mask_t] = labeled_mask[src_idx, paste_mask_t]

            # logits/confidence
            mix_unlabeled_logits[i, paste_mask_t] = labeled_logits[src_idx, paste_mask_t]

    # 3) second stage: paste mixed unlabeled regions into shuffled unlabeled samples
    for i in range(unlabeled_image.shape[0]):
        paste_mask = _build_shape_aware_region(
            unlabeled_img_i=unlabeled_image[i],
            unlabeled_logits_i=unlabeled_logits[i],
            image_confidence=lst_confidences[i]
        )

        paste_mask_t = torch.from_numpy(paste_mask).to(unlabeled_image.device)
        src_idx = u_rand_index[i]

        unlabeled_image[i, :, paste_mask_t] = mix_unlabeled_image[src_idx, :, paste_mask_t]
        unlabeled_mask[i, paste_mask_t] = mix_unlabeled_target[src_idx, paste_mask_t]
        unlabeled_logits[i, paste_mask_t] = mix_unlabeled_logits[src_idx, paste_mask_t]

    del mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, labeled_logits

    return unlabeled_image, unlabeled_mask, unlabeled_logits