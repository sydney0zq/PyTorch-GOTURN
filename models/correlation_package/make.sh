#!/usr/bin/env bash
TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src

echo "Compiling correlation kernels by nvcc..."

rm correlation_cuda_kernel.o
#rm -r ../_ext

/usr/local/cuda/bin/nvcc -c -o correlation_cuda_kernel.o correlation_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
