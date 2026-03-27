tport=52009
ngpu=1
ROOT=$HOME/Projects/AugSeg

torchrun \
  --nproc_per_node=${ngpu} \
  --master_port=${tport} \
  $ROOT/train_semi.py \
  --config=$ROOT/exps/zrun_citys/citys_semi744/config_semi.yaml \
  --seed 2 \
  --port ${tport}