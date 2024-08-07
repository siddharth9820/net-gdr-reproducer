#!/bin/bash
#SBATCH -p batch
#SBATCH -t 00:40:00

module load PrgEnv-cray
rocm_version="5.7.0"
module load amd-mixed/${rocm_version}
module load cray-mpich/8.1.28
module load cpe/23.12
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load libtool
export ROCM_PATH="/opt/rocm-${rocm_version}/"
export MPICH_GPU_SUPPORT_ENABLED=1

## CHANGE THIS
PROJ_NAME="csc569"
export WRKSPC="/lustre/orion/${PROJ_NAME}/scratch/${USER}/gordon-bell/reproducer-net-gdr"
ENV_NAME="gordon-bell-venv"
ENV_LOC="$WRKSPC/$ENV_NAME"

# activate venv
. ${ENV_LOC}/bin/activate

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CRAY_MPICH_ROOTDIR/gtl/lib"
## some RCCL env variables
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=3
#export NCCL_MIN_NRINGS=4
#export NCCL_DEBUG="WARN"
#RCCL version 2.17.1+hip5.7 HEAD:cbbb3d8+0
# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export OMP_NUM_THREADS=7 
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WRKSPC/aws-ofi-rccl/lib"

run_cmd="srun -N $NNODES -n $GPUS -c7 --gpus-per-task=1 --gpu-bind=closest ./get_rank_from_slurm.sh python -u benchmark_mlp.py" 

echo $run_cmd
eval $run_cmd
set +x

