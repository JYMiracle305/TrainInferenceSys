# 复现llm.c过程

链接：git@github.com:karpathy/llm.c.git

## 提示

在编译过程会有以下提示，指示当前环境中配置的情况，可以根据执行的步骤中进行查漏补缺

---------------------------------------------
✓ cuDNN found, will run with flash-attention

✓ OpenMP found

✓ NCCL found, OK to train with multiple GPUs

✓ MPI enabled

✓ nvcc found, including GPU/CUDA support

---------------------------------------------

## quick start (1 GPU, fp32 only)
执行命令
1. 下载预处理的.bin文件
```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh

```

由于hugging face服务器的原因，可能下载失败，可以通过以下命令重新生成.bin
```bash
pip install -r requirements.txt  #安装必要依赖  pytorch==2.6.0
python dev/data/tinyshakespeare.py
python train_gpt2.py   #如果失败，添加一个设置huggingface镜像的语句  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

2. 编译+训练
```bash
make train_gpt2fp32cu
./train_gpt2fp32cu
```

## quick start (CPU)
执行命令

1. 预置bin文件同上

2. 编译+训练
```bash
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

## 测试CPU
```bash
make test_gpt2
./test_gpt2
```

##
```bash
# fp32 test (cudnn not supported)
make test_gpt2cu PRECISION=FP32 && ./test_gpt2cu
# 需要在makefile中指定MPI的路径

# mixed precision cudnn test
make test_gpt2cu USE_CUDNN=1 && ./test_gpt2cu

# 使用cudnn的需要cudnn 8.9.3以上版本
# 使用cudnn的时候，需要配置cudnn frontend，clone仓库（git@github.com:NVIDIA/cudnn-frontend.git），到本地后在makefile中配置CUDNN_FRONTEND_PATH 指向include目录
```