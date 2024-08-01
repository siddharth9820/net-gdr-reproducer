#!/bin/bash

module load PrgEnv-cray
rocm_version="5.7.0"
module load amd-mixed/${rocm_version}
module load cray-mpich/8.1.28
module load cpe/23.12
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load libtool

export ROCM_PATH="/opt/rocm-${rocm_version}/"


## CHANGE THIS
PROJ_NAME="csc569"
export WRKSPC="/lustre/orion/${PROJ_NAME}/scratch/${USER}/gordon-bell/reproducer-net-gdr"
mkdir -p $WRKSPC
cd $WRKSPC
ENV_NAME="gordon-bell-venv"
ENV_LOC="$WRKSPC/$ENV_NAME"

# Setup Virtual Environment
echo "Setting up Virtual Environment"
python -m venv ${ENV_LOC} --system-site-packages
. ${ENV_LOC}/bin/activate

# PyTorch
echo "Installing PyTorch"
if [ "${rocm_version}" == 5.6.0  ]; then
	pip install --force-reinstall /lustre/orion/world-shared/stf007/msandov1/wheels/TorchROCm5.6/torch-2.1.2-cp310-cp310-linux_x86_64.whl
elif [ "${rocm_version}" == 6.0.0  ]; then
	pip3 install torch  --index-url https://download.pytorch.org/whl/rocm6.0
elif [ "${rocm_version}" == 5.7.0  ]; then
	pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
fi

# RCCL plugin
echo "Installing RCCL Plugin"
git clone --recursive --depth=1 https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
cd aws-ofi-rccl
libfabric_path=/opt/cray/libfabric/1.15.2.0
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
CC=cc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PWD --with-hip=/opt/rocm-$rocm_version --with-mpi=$MPICH_DIR
make
make install
cd ..
tar -cvzf aws-ofi-rccl.tar.gz aws-ofi-rccl/
echo "export WRKSPC_GB=/lustre/orion/$PROJ_NAME/scratch/$USER/gordon-bell" >> ~/.bashrc


git clone git@github.com:axonn-ai/axonn.git
cd axonn
git checkout add-timers-gordon-bell
pip install -e .



