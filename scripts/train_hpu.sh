#!/bin/bash

export PT_HPU_LAZY_MODE=0
export LOCAL_RANK=hpu

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
--num_steps=150
"

torchrun \
    --nnodes=$NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port="12234" \
    ../main_training.py \
${MODEL_ARGS}
