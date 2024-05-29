#!/bin/bash

#export GPU_MIGRATION_LOG_LEVEL=1
export PT_HPU_LAZY_MODE=0

datastore_path=/root

MODEL_ARGS="\
--use_dummy_dataset=True
--data_path=/software/users/hlahkar/my-copy-c4
--ckpt_load_path=$datastore_path/pretrain/ckpt
--ckpt_save_path=$datastore_path/pretrain/ckpt
--fsdp_activation_checkpointing=False
--selective_checkpointing=0
--sharding_strategy=hsdp
--low_cpu_fsdp=False
--batch_size=2
--report_interval=1
--checkpoint_interval=20000
--use_torch_compile=True
--use_profiler=False
--use_hpu=True
"

#python -m torch.distributed.launch \
#--nproc_per_node=8 \
#../main_training.py \
#${MODEL_ARGS}

torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=8 \
    --master_addr=10.233.151.164 \
    --master_port="12234" \
    ../main_training.py \
    ${MODEL_ARGS}

#mpirun -n 8 --bind-to core --map-by socket:PE=10 --rank-by core --report-bindings --allow-run-as-root \
#python ../main_training.py \
#${MODEL_ARGS}
