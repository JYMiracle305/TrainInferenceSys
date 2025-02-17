import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
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

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 训练函数
def train(data_path, model_name="gpt2", epochs=3, batch_size=8, device="cuda"):
    print("模型名称:", model_name)
    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 数据加载
    dataset = TextDataset(tokenizer, data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * epochs)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 每个epoch保存一次模型
        model.save_pretrained(f"model_epoch_{epoch}")

# 主程序
if __name__ == "__main__":
    data_path = "data/train.txt"  # 替换为你的训练数据路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(data_path, device=device)