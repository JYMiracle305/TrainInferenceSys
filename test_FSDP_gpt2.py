import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim import OSS

# 定义数据集
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenized_text = self.tokenizer.encode(text, return_tensors="pt")[0]
        return tokenized_text

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = (idx + 1) * self.block_size
        return self.data[start:end]

# 初始化模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 包装模型为 FSDP
model = FSDP(model)

# 数据加载
file_path = "data/train.txt"  # 替换为你的训练数据文件路径
dataset = TextDataset(tokenizer, file_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 优化器
optimizer = OSS(params=model.parameters(), optim=Adam, lr=1e-5)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # 训练 3 个 epoch
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# 保存模型
torch.save(model.state_dict(), "gpt2_fairscale.pth")