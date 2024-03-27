#!/bin/bash
#SBATCH --account=answerai
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-node=1
#SBATCH --mem=256gb
#SBATCH --cpus-per-gpu=12
#SBATCH --job-name=eval-ft-models-single-gpu
#SBATCH --output=sbatch_outputs/%x_%j.out

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

export MASTER_PORT=12342
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_GPUS_PER_NODE))

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "Starting python script"


SHARED_VOLUME_DIR=/weka/home-$(whoami)
source $SHARED_VOLUME_DIR/py_venvs/vllm-py311/bin/activate

# nccl
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export OMPI_MCA_mtl_base_verbose=1
export FI_PROVIDER=efa
export NCCL_TREE_THRESHOLD=0

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_TIMEOUT=600000 # Set the timeout to 10 minutes (60000 milliseconds)
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
echo "Using python from $(which python)"
echo "Using torch from $(python -c 'import torch; print(torch.__file__)')"
echo "Using torch cuda from $(python -c 'import torch; print(torch.version.cuda)')"
echo "Using nccl from $(python -c 'import torch; print(torch.cuda.nccl.version())')"

# print cuda home
echo "CUDA_HOME=$CUDA_HOME"

# GLOBAL_BATCH_SIZE=64
MAX_BATCH_SIZE=1
# 32 / world size
GRAD_ACCUM_STEPS=16


# Full finetune post quantization.
# QLORA
# QDORA
# Q-LLAMA-Pro

# srun python $SHARED_VOLUME_DIR/git/fsdp_qlora/evaluate.py \
# --eval_type full_post_quantized \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /weka/home-keremturgutlu/models/ \
# --trained_model_dir llama-7b-orca-math-10k-full \
# --save_path /weka/home-keremturgutlu/git/fsdp_qlora/eval_results/10k-full-post-quantize.json

# srun python $SHARED_VOLUME_DIR/git/fsdp_qlora/evaluate.py \
# --eval_type qlora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /weka/home-keremturgutlu/models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-qlora \
# --save_path /weka/home-keremturgutlu/git/fsdp_qlora/eval_results/10k-qlora.json

# srun python $SHARED_VOLUME_DIR/git/fsdp_qlora/evaluate.py \
# --eval_type bnb_dora \
# --model_name meta-llama/Llama-2-7b-hf \
# --models_dir /weka/home-keremturgutlu/models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-qdora \
# --save_path /weka/home-keremturgutlu/git/fsdp_qlora/eval_results/10k-bnb-dora.json

# srun python $SHARED_VOLUME_DIR/git/fsdp_qlora/evaluate.py \
# --eval_type bnb_llama_pro \
# --model_name meta-llama/Llama-2-7b-hf \
# --llama_pro_path /weka/home-keremturgutlu/models/meta-llama/Llama-2-7b-hf_blk_exp-32-35/ \
# --models_dir /weka/home-keremturgutlu/models/ \
# --trained_model_dir llama-7b-orca-math-10k-bnb-llama-pro \
# --save_path /weka/home-keremturgutlu/git/fsdp_qlora/eval_results/10k-bnb-llama-pro.json