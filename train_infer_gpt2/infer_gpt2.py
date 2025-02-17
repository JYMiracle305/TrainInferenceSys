import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import os

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512, mode="train"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenized_text = self.tokenizer.encode(text, add_special_tokens=False)
        if self.mode == "train":
            return [
                tokenized_text[i : i + self.block_size]
                for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size)
            ]
        elif self.mode == "validation":
            return [
                tokenized_text[i : i + self.block_size]
                for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size * 2)
            ]  # 可以调整验证数据的采样方式

def train(data_path, val_path, model_name="gpt2", epochs=3, batch_size=8, device="cuda"):
    print("模型名称:", model_name)
    # 加载模型和分词器
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 数据加载
    train_dataset = TextDataset(tokenizer, data_path, mode="train")
    val_dataset = TextDataset(tokenizer, val_path, mode="validation")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * epochs)

    # 训练循环
    model.train()
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                outputs = model(batch, labels=batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

        # 每个epoch保存一次模型
        model.save_pretrained(f"model_epoch_{epoch}")

def generate_text(model, tokenizer, prompt, max_length=50, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_text = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated_text[0], skip_special_tokens=True)

# main
if __name__ == "__main__":
    data_path = "data/train.txt"  # 训练数据路径
    val_path = "data/val.txt"    # 验证数据路径
    if (torch.cuda.is_available()):
        print("torch cuda!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练模型
    train(data_path, val_path, device=device)

    # 加载模型并生成文本
    model = GPT2LMHeadModel.from_pretrained("model_epoch_1").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "今天天气不错"
    generated_text = generate_text(model, tokenizer, prompt, max_length=100, device=device)
    print("Generated Text:", generated_text)
