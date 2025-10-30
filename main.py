"""
使用完整打字数据训练模型
支持：拼音过程 + 汉字 + 英文 + 删除
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TypingStep:
    action: str  # 'ime_update', 'type', 'delete'
    content: str
    duration: float


class TypingSequence:
    def __init__(self, target_text: str):
        self.target_text = target_text
        self.steps: List[TypingStep] = []
    
    def add_step(self, action: str, content: str, duration: float):
        self.steps.append(TypingStep(action, content, duration))
    
    def __len__(self):
        return len(self.steps)


class SmartVocab:
    def __init__(self):
        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.special_tokens = [self.PAD, self.SOS, self.EOS]
        self.token2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.next_idx = len(self.special_tokens)
    
    def add_token(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = self.next_idx
            self.idx2token[self.next_idx] = token
            self.next_idx += 1
    
    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.PAD])
    
    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, self.PAD)
    
    def __len__(self):
        return len(self.token2idx)


def load_typing_data(json_path: str) -> List[TypingSequence]:
    """
    加载完整打字数据
    
    格式:
    {
        "target": "今天天气很好",
        "typing_log": [
            {"time": 0.2, "action": "ime_update", "content": "jt"},
            {"time": 0.5, "action": "type", "content": "今天", "method": "ime"},
            {"time": 1.2, "action": "delete", "deleted": "天"}
        ]
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = []
    for record in data:
        target = record['target']
        typing_log = record['typing_log']
        
        sequence = TypingSequence(target)
        
        last_time = 0.0
        for log_entry in typing_log:
            action = log_entry['action']
            timestamp = log_entry['time']
            duration = timestamp - last_time
            
            if action == 'ime_update':
                # 拼音过程
                content = log_entry['content']
                sequence.add_step('IME', content, duration)
                
            elif action == 'type':
                # 汉字/英文输入
                content = log_entry['content']
                sequence.add_step('TYPE', content, duration)
                
            elif action == 'delete':
                # 删除
                deleted = log_entry.get('deleted', '')
                sequence.add_step('DELETE', deleted, duration)
            
            last_time = timestamp
        
        sequence.add_step('EOS', '', 0.0)
        sequences.append(sequence)
    
    return sequences


class TypingDataset(Dataset):
    def __init__(self, sequences: List[TypingSequence], vocab: SmartVocab):
        self.sequences = sequences
        self.vocab = vocab
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 编码目标
        target_ids = [self.vocab.encode(sequence.target_text)]
        
        # 编码动作序列
        action_ids = [self.vocab.encode(self.vocab.SOS)]
        durations = [0.0]
        
        for step in sequence.steps:
            if step.action == 'IME':
                action_ids.append(self.vocab.encode(f'IME:{step.content}'))
            elif step.action == 'TYPE':
                action_ids.append(self.vocab.encode(f'TYPE:{step.content}'))
            elif step.action == 'DELETE':
                action_ids.append(self.vocab.encode('DELETE'))
            elif step.action == 'EOS':
                action_ids.append(self.vocab.encode(self.vocab.EOS))
            
            durations.append(step.duration)
        
        return {
            'target_ids': torch.LongTensor(target_ids),
            'action_ids': torch.LongTensor(action_ids),
            'durations': torch.FloatTensor(durations),
        }


def collate_fn(batch):
    max_target_len = max(len(item['target_ids']) for item in batch)
    max_action_len = max(len(item['action_ids']) for item in batch)
    
    target_ids_batch = []
    action_ids_batch = []
    durations_batch = []
    target_masks = []
    action_masks = []
    
    for item in batch:
        target_len = len(item['target_ids'])
        target_pad = torch.zeros(max_target_len - target_len, dtype=torch.long)
        target_ids_batch.append(torch.cat([item['target_ids'], target_pad]))
        target_masks.append(torch.cat([torch.ones(target_len), torch.zeros(max_target_len - target_len)]))
        
        action_len = len(item['action_ids'])
        action_pad = torch.zeros(max_action_len - action_len, dtype=torch.long)
        action_ids_batch.append(torch.cat([item['action_ids'], action_pad]))
        action_masks.append(torch.cat([torch.ones(action_len), torch.zeros(max_action_len - action_len)]))
        
        duration_pad = torch.zeros(max_action_len - action_len)
        durations_batch.append(torch.cat([item['durations'], duration_pad]))
    
    return {
        'target_ids': torch.stack(target_ids_batch),
        'action_ids': torch.stack(action_ids_batch),
        'durations': torch.stack(durations_batch),
        'target_masks': torch.stack(target_masks),
        'action_masks': torch.stack(action_masks),
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class TypingTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.action_head = nn.Linear(d_model, vocab_size)
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, target_ids, action_ids, target_mask=None, action_mask=None):
        target_emb = self.embedding(target_ids) * np.sqrt(self.d_model)
        target_emb = self.pos_encoder(target_emb)
        target_padding_mask = ~target_mask.bool() if target_mask is not None else None
        memory = self.encoder(target_emb, src_key_padding_mask=target_padding_mask)
        
        action_emb = self.embedding(action_ids) * np.sqrt(self.d_model)
        action_emb = self.pos_encoder(action_emb)
        
        action_len = action_ids.size(1)
        causal_mask = torch.triu(torch.ones(action_len, action_len, device=action_ids.device), diagonal=1).bool()
        action_padding_mask = ~action_mask.bool() if action_mask is not None else None
        
        output = self.decoder(action_emb, memory, tgt_mask=causal_mask, 
                            tgt_key_padding_mask=action_padding_mask,
                            memory_key_padding_mask=target_padding_mask)
        
        action_logits = self.action_head(output)
        duration_pred = self.duration_head(output).squeeze(-1)
        return action_logits, duration_pred
    
    @torch.no_grad()
    def generate(self, target_text: str, vocab: SmartVocab, max_steps: int = 100,
                 temperature: float = 0.8, device: str = 'cpu') -> TypingSequence:
        self.eval()
        
        target_ids = torch.LongTensor([[vocab.encode(target_text)]]).to(device)
        target_emb = self.embedding(target_ids) * np.sqrt(self.d_model)
        target_emb = self.pos_encoder(target_emb)
        memory = self.encoder(target_emb)
        
        sequence = TypingSequence(target_text)
        generated_ids = [vocab.encode(vocab.SOS)]
        
        for step in range(max_steps):
            generated_ids_safe = [min(max(0, id), self.vocab_size-1) for id in generated_ids]
            action_ids = torch.LongTensor([generated_ids_safe]).to(device)
            action_emb = self.embedding(action_ids) * np.sqrt(self.d_model)
            action_emb = self.pos_encoder(action_emb)
            
            action_len = action_ids.size(1)
            causal_mask = torch.triu(torch.ones(action_len, action_len, device=device), diagonal=1).bool()
            output = self.decoder(action_emb, memory, tgt_mask=causal_mask)
            
            next_action_logits = output[0, -1] / temperature
            if next_action_logits.size(0) > self.vocab_size:
                next_action_logits = next_action_logits[:self.vocab_size]
            
            next_action_probs = F.softmax(next_action_logits, dim=-1)
            next_action_id = torch.multinomial(next_action_probs, 1).item()
            next_action_id = min(max(0, next_action_id), self.vocab_size - 1)
            next_action_token = vocab.decode(next_action_id)
            next_duration = self.duration_head(output[0, -1:]).item()
            
            if next_action_token == vocab.EOS:
                sequence.add_step('EOS', '', next_duration)
                break
            elif next_action_token.startswith('IME:'):
                content = next_action_token[4:]
                sequence.add_step('IME', content, next_duration)
            elif next_action_token.startswith('TYPE:'):
                content = next_action_token[5:]
                sequence.add_step('TYPE', content, next_duration)
            elif next_action_token == 'DELETE':
                sequence.add_step('DELETE', '', next_duration)
            
            generated_ids.append(next_action_id)
        
        return sequence


class Trainer:
    def __init__(self, model: TypingTransformer, vocab: SmartVocab,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', lr: float = 3e-4):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-6)
        self.action_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = total_action_loss = total_duration_loss = 0
        
        for batch in dataloader:
            target_ids = batch['target_ids'].to(self.device)
            action_ids = batch['action_ids'].to(self.device)
            durations = batch['durations'].to(self.device)
            target_masks = batch['target_masks'].to(self.device)
            action_masks = batch['action_masks'].to(self.device)
            
            action_logits, duration_pred = self.model(target_ids, action_ids[:, :-1], target_masks, action_masks[:, :-1])
            
            action_loss = self.action_criterion(action_logits.reshape(-1, action_logits.size(-1)), action_ids[:, 1:].reshape(-1))
            
            valid_mask = action_masks[:, 1:] * (durations[:, 1:] > 0)
            duration_loss = (((duration_pred - durations[:, 1:]) ** 2) * valid_mask).sum() / valid_mask.sum() if valid_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
            
            loss = action_loss + 0.1 * duration_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_duration_loss += duration_loss.item()
        
        self.scheduler.step()
        return {
            'total_loss': total_loss / len(dataloader),
            'action_loss': total_action_loss / len(dataloader),
            'duration_loss': total_duration_loss / len(dataloader),
        }
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
        }, path)


def main():
    print("=" * 70)
    print("完整打字行为模型训练")
    print("=" * 70 + "\n")
    
    # 加载数据
    try:
        print("📂 加载数据: typing_data_with_pinyin.json")
        sequences = load_typing_data('typing_data_with_pinyin.json')
        print(f"✅ 加载了 {len(sequences)} 条打字记录\n")
    except FileNotFoundError:
        print("❌ 找不到 typing_data_with_pinyin.json")
        print("\n请先运行数据收集器:")
        print("  python data_collector_ultimate.py")
        print("  收集5-10条数据后再训练\n")
        return
    
    # 构建词汇表
    vocab = SmartVocab()
    for seq in sequences:
        vocab.add_token(seq.target_text)  # 目标文本
        for step in seq.steps:
            if step.action == 'IME':
                vocab.add_token(f'IME:{step.content}')
            elif step.action == 'TYPE':
                vocab.add_token(f'TYPE:{step.content}')
            elif step.action == 'DELETE':
                vocab.add_token('DELETE')
    
    print(f"📚 词汇表大小: {len(vocab)}")
    print(f"   包含:")
    print(f"   - 拼音动作 (IME:xxx)")
    print(f"   - 输入动作 (TYPE:xxx)")
    print(f"   - 删除动作 (DELETE)\n")
    
    # 创建数据集
    dataset = TypingDataset(sequences, vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 设备: {device}\n")
    
    model = TypingTransformer(vocab_size=len(vocab), d_model=256, nhead=8, 
                              num_encoder_layers=4, num_decoder_layers=4,
                              dim_feedforward=1024, dropout=0.1)
    
    trainer = Trainer(model, vocab, device, lr=3e-4)
    
    # 训练
    print("🚀 开始训练...\n")
    num_epochs = 100
    
    for epoch in range(num_epochs):
        metrics = trainer.train_epoch(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {metrics['total_loss']:.4f} | "
                  f"Action: {metrics['action_loss']:.4f} | "
                  f"Time: {metrics['duration_loss']:.4f}")
    
    print("\n✅ 训练完成!\n")
    
    # 测试生成
    test_texts = ["今天天气很好", "我喜欢编程"]
    
    for test_text in test_texts:
        print(f"\n{'='*70}")
        print(f"测试: {test_text}")
        print(f"{'-'*70}")
        
        sequence = model.generate(target_text=test_text, vocab=vocab, temperature=0.9, device=device)
        
        current_time = 0.0
        for step in sequence.steps:
            current_time += step.duration
            if step.action == 'IME':
                print(f"  [{current_time:5.2f}s] 📝 拼音: {step.content}")
            elif step.action == 'TYPE':
                print(f"  [{current_time:5.2f}s] ✅ 输入: {step.content}")
            elif step.action == 'DELETE':
                print(f"  [{current_time:5.2f}s] ❌ 删除")
            elif step.action == 'EOS':
                print(f"  [{current_time:5.2f}s] 🏁 结束")
    
    # 保存模型
    save_path = 'typing_model.pth'
    trainer.save(save_path)
    print(f"\n💾 模型已保存到当前目录: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()