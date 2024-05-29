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
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=10.233.151.164 \
    --master_port="12234" \
    ../main_training.py \
    ${MODEL_ARGS}

#mpirun --hostfile hostnames --allow-run-as-root --tag-output --merge-stderr-to-stdout \
#--bind-to core --map-by slot:PE=10 --np 16 --mca plm_rsh_args -p3022 \
#--mca btl_tcp_if_include eth1 --prefix /opt/amazon/openmpi --report-bindings \
#--output-filename /tmp/bert_pt_log --rank-by core \
#-x MASTER_PORT=12345 -x MASTER_ADDR=10.233.236.231 \
#-x HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
#-x PYTHONPATH=/usr/lib/habanalabs:/root/repos/event_tests_plugin/:/root/repos/tensorflow-training-tests/:/root/repos/model_garden:/root/repos/model_garden/internal:/root/repos/gc_tools:/root/repos/tensorflow-training/:/usr/lib/habanalabs/:/root/repos/event_tests_plugin:/root/repos/unilm:$PYTHONPATH -x GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/lib/habanalabs:/opt/amazon/openmpi/lib/ -x PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/amazon/openmpi/bin/ -x HABANA_LOGS=/root/repos/results/habana_logs \
#python ../main_training.py \
#${MODEL_ARGS}
