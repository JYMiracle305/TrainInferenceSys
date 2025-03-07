import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP2
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig
from datasets import load_dataset
import functools
import time
from faker import Faker
import argparse
from torch.distributed.fsdp import CPUOffload
import tracemalloc

# 初始化分布式环境
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # 模型参数存储为 float16
    reduce_dtype=torch.bfloat16,    # 梯度规约使用 float16
    buffer_dtype=torch.bfloat16     # 缓冲区（如BatchNorm）使用 float16
)
# 加载模型并包装为FSDP
def load_model(model_name, use_checkpointing=True):
    config = AutoConfig.from_pretrained(model_name)
    model = LlamaForCausalLM(config)
    
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e8
    )

    cpu_offload = CPUOffload(offload_params=os.environ.get("CPU_OFFLOAD", "false").lower() == "true")

    # FSDP包装模型（分片参数、梯度、优化器状态）
    model = FSDP(
        module=model,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=auto_wrap_policy if os.environ.get("AUTO_WRAP_POLICY", "false").lower() == "true" else None,
        cpu_offload=cpu_offload   # 启用显存卸载
    )

    # model = FSDP2(
    #     model,
    #     mixed_precision=torch.float16
    # )

    # print("Auto wrap policy:", auto_wrap_policy)

    return model

class MyCustomDataset(Dataset):
    def __init__(self, tokenizer, max_length, num_samples=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.fake = Faker()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 自动生成文本数据
        text = self.fake.text(max_nb_chars=self.max_length)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

# 数据预处理
def prepare_dataloader(tokenizer, batch_size=4, seq_length=512, num_samples=1000):
    dataset = MyCustomDataset(
        tokenizer=tokenizer,
        max_length=seq_length,
        num_samples=num_samples
    )
    
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    return dataloader

# 训练循环
def train(batch_size, gradient_accumulation_steps):
    rank, world_size, local_rank = setup_distributed()
    
    # 加载模型和分词器
    model_name = "../model/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token
    
    model = load_model(model_name)
    model.train()
    # model.to(local_rank)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 混合精度梯度缩放
    scaler = GradScaler(enabled=True)
    
    # 数据加载器
    dataloader = prepare_dataloader(tokenizer, batch_size=batch_size, seq_length=2048, num_samples=1000)
    
    # 训练参数
    max_steps = 1000
    
    tracemalloc.start()

    # 训练循环
    step = 0
    for epoch in range(1):
        print(f"epoch {epoch}:")
        dataloader.sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            inputs = {k: v.to(local_rank) for k, v in batch.items()}
            
            start_time = time.time()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            end_time = time.time()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                
                # 计算吞吐量
                batch_tokens = inputs["input_ids"].numel()  # 当前批次的 token 数量
                elapsed_time = end_time - start_time  # 处理时间
                throughput = batch_tokens / elapsed_time  # token/s

                if rank == 3:
                    print(f"rank {rank}, write throughput")
                    with open("throughput.log", "a") as f:
                        f.write(f"{throughput:.2f}\n")

                    #memory_usage = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)  # 转换为 GB
                    peak_memory = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
                    with open("memory.log", "a") as f:
                        f.write(f"{peak_memory:.2f}\n")
                print(f"rank: {rank}, Step {step}, Loss: {loss.item() * gradient_accumulation_steps}, Throughput: {throughput:.2f} token/s")
                
                if step >= max_steps:
                    break
    tracemalloc.stop()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    train(batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps)

    print("train successfully!")