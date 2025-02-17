import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel import initialize_model_parallel
from fairscale.optim.oss import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import os

# 数据集类
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenized_text = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenized_text[i : i + self.block_size]
            for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# 初始化分布式环境
def init_distributed():
    dist.init_process_group(backend="nccl")
    initialize_model_parallel(1, 1)  # 1 GPU per model parallel group, 1 data parallel group

# 主训练函数
def train(rank, world_size, data_path, model_name="gpt2", epochs=3, batch_size=8):
    # 初始化分布式环境
    init_distributed()
    torch.cuda.set_device(rank)

    # 加载模型和词分器
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 数据加载
    dataset = TextDataset(tokenizer, data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 优化器和学习率调度器
    optimizer= OSS(params=model.parameters(), optim=AdamW, lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * epochs)
    scaler = ShardedGradScaler()

    # 分布式模型包装
    model = DDP(model, device_ids=[rank])

    # 训练循环
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = batch.to(rank)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # 混合精度训练
                outputs = model(batch, labels=batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 10 == 0 and rank == 0:  # 打印日志
                print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if rank == 0:  # 保存模型
            model.module.save_pretrained(f"model_epoch_{epoch}")

    dist.destroy_process_group()

# 启动训练
if __name__ == "__main__":
    data_path = "path/to/your/training_data.txt"  # 替换为你的训练数据路径
    world_size = torch.cuda.device_count()  # 使用所有可用的GPU
    torch.multiprocessing.spawn(train, args=(world_size, data_path), nprocs=world_size, join=True)