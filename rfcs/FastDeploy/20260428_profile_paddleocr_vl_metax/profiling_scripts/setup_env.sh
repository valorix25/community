#!/bin/sh
export MACA_PATH=/opt/maca

if [ ! -d ${HOME}/cu-bridge ]; then
  `${MACA_PATH}/tools/cu-bridge/tools/pre_make`
fi

export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export MACA_VISIBLE_DEVICES="0"
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection