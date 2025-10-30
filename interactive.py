"""
交互式打字模拟器
用法: python interact.py typing_model.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import List


# ========== 终端显示宽度工具 ==========
def char_cell_width(ch: str) -> int:
    """
    计算单个字符在等宽终端中的“显示列宽”：
    - 全角/宽字符(F/W)：2
    - 组合符(Mn)、回车换行等：0
    - 其他：1
    """
    if not ch:
        return 0
    if ch in ('\n', '\r'):
        return 0
    cat = unicodedata.category(ch)
    if cat == 'Mn':  # Combining mark
        return 0
    eaw = unicodedata.east_asian_width(ch)
    return 2 if eaw in ('F', 'W') else 1


def str_cell_width(s: str) -> int:
    return sum(char_cell_width(ch) for ch in s or '')


def backspace_cells(n: int):
    """
    回退 n 个“显示列”。每一列用 '\b \b' 覆盖清除。
    """
    for _ in range(max(0, n)):
        print('\b \b', end='', flush=True)


# ========== 原有数据结构 ==========
@dataclass
class TypingStep:
    action: str
    content: str
    duration: float


class TypingSequence:
    def __init__(self, target_text: str):
        self.target_text = target_text
        self.steps: List[TypingStep] = []
    
    def add_step(self, action: str, content: str, duration: float):
        self.steps.append(TypingStep(action, content, duration))


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
            next_duration = max(0.05, self.duration_head(output[0, -1:]).item())
            
            if next_action_token == vocab.EOS:
                sequence.add_step('EOS', '', next_duration)
                break
            elif isinstance(next_action_token, str) and next_action_token.startswith('IME:'):
                content = next_action_token[4:]
                sequence.add_step('IME', content, next_duration)
            elif isinstance(next_action_token, str) and next_action_token.startswith('TYPE:'):
                content = next_action_token[5:]
                sequence.add_step('TYPE', content, next_duration)
            elif next_action_token == 'DELETE':
                sequence.add_step('DELETE', '', next_duration)
            
            generated_ids.append(next_action_id)
        
        return sequence


def load_model(model_path: str):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab = checkpoint['vocab']
    
    model = TypingTransformer(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


def animate_typing(sequence: TypingSequence, speed_factor: float = 1.0):
    """实时演示打字过程 - 模拟真实打字效果（修复：按显示列宽删除拼音/字符 + EOS清理）"""
    print(f"\n目标: {sequence.target_text}")
    print("打字过程: ", end='', flush=True)
    
    current_output = ""
    pinyin_buffer_text = ""   # 已打印的拼音文本
    pinyin_buffer_cols = 0    # 已打印拼音的“显示列数”
    total_time = 0.0
    step_records = []  # 记录每一步的详细信息（仅用于表格）
    
    for step in sequence.steps:
        # 等待（模拟真实打字时间）
        sleep_time = max(0.0, float(step.duration)) * speed_factor
        if sleep_time > 0:
            sleep_time = min(sleep_time, 0.25 * speed_factor)  # 限制单步等待
            time.sleep(sleep_time)
        
        total_time += step.duration
        
        if step.action == 'IME':
            # 显示拼音（临时显示，会被删除）
            safe = (step.content or '').replace('\n', '').replace('\r', '').replace('\t', '')
            if safe:
                print(safe, end='', flush=True)
                pinyin_buffer_text += safe
                pinyin_buffer_cols += str_cell_width(safe)
            step_records.append({
                'time': total_time,
                'action': 'IME拼音',
                'content': safe,
                'buffer': pinyin_buffer_text,
                'output': current_output
            })
        
        elif step.action == 'TYPE':
            # 如果有拼音缓冲，按“显示列宽”删除拼音
            deleted_pinyin = pinyin_buffer_text
            if pinyin_buffer_cols > 0:
                backspace_cells(pinyin_buffer_cols)
                pinyin_buffer_text = ""
                pinyin_buffer_cols = 0
            
            # 输入汉字/英文（可多字符）
            safe = (step.content or '').replace('\n', '').replace('\r', '').replace('\t', '')
            if safe:
                current_output += safe
                print(safe, end='', flush=True)
            step_records.append({
                'time': total_time,
                'action': 'TYPE输入',
                'content': safe,
                'deleted_pinyin': deleted_pinyin if deleted_pinyin else '-',
                'output': current_output
            })
        
        elif step.action == 'DELETE':
            # 删除一个“当前输出”的字符 —— 也按显示列宽删除
            deleted_char = current_output[-1] if current_output else ''
            if current_output:
                current_output = current_output[:-1]
                backspace_cells(char_cell_width(deleted_char))
            step_records.append({
                'time': total_time,
                'action': 'DELETE删除',
                'content': deleted_char,
                'output': current_output
            })
        
        elif step.action == 'EOS':
            # >>> 新增：若EOS时仍有拼音残留，先删干净再结束 <<<
            if pinyin_buffer_cols > 0:
                backspace_cells(pinyin_buffer_cols)
                pinyin_buffer_text = ""
                pinyin_buffer_cols = 0
            step_records.append({
                'time': total_time,
                'action': 'EOS结束',
                'content': '-',
                'output': current_output
            })
            break

    # >>> 兜底：循环结束后再次清理可能残留的拼音 <<<
    if pinyin_buffer_cols > 0:
        backspace_cells(pinyin_buffer_cols)
        pinyin_buffer_text = ""
        pinyin_buffer_cols = 0
    
    print()  # 换行
    print(f"\n最终输出: {current_output}")
    matched = current_output == sequence.target_text
    print(f"是否匹配: {'✅' if matched else '❌'}")
    
    # 显示详细的步骤历史记录
    print(f"\n{'='*80}")
    print(f"📝 打字步骤详细记录 (共 {len(step_records)} 步)")
    print(f"{'='*80}")
    print(f"{'步骤':<6} {'时间':<8} {'动作':<12} {'内容':<15} {'当前输出':<20}")
    print(f"{'-'*80}")
    
    for i, record in enumerate(step_records, 1):
        time_str = f"{record['time']:.2f}s"
        content = record['content']
        output = record['output'] if record['output'] else '(空)'
        
        if record['action'] == 'TYPE输入':
            deleted = record.get('deleted_pinyin', '-')
            print(f"{i:<6} {time_str:<8} {record['action']:<12} {content:<15} {output:<20}")
            if deleted != '-':
                print(f"{'':6} {'':8} {'→删除拼音':<12} {deleted:<15}")
        else:
            print(f"{i:<6} {time_str:<8} {record['action']:<12} {content:<15} {output:<20}")
    
    print(f"{'='*80}\n")
    
    # 返回统计信息
    return {
        'final_output': current_output,
        'matched': matched,
        'total_time': total_time,
        'step_count': len(sequence.steps)
    }


def interactive_mode(model, vocab, device):
    """交互式模式"""
    print("\n" + "="*70)
    print("🎮 交互式打字模拟器")
    print("="*70)
    print("\n命令:")
    print("  - 输入文本: 直接输入想要模拟的文本")
    print("  - 'speed X': 设置播放速度 (0.5=慢速, 1.0=正常, 2.0=快速)")
    print("  - 'temp X': 设置温度 (0.5-1.5, 越低越确定)")
    print("  - 'quit': 退出\n")
    
    speed = 1.0
    temperature = 0.8
    
    while True:
        try:
            user_input = input("📝 输入文本 (或命令): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 再见！")
                break
            
            if user_input.startswith('speed '):
                try:
                    speed = float(user_input.split()[1])
                    print(f"✅ 播放速度设为: {speed}x")
                except:
                    print("❌ 格式错误，用法: speed 1.0")
                continue
            
            if user_input.startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"✅ 温度设为: {temperature}")
                except:
                    print("❌ 格式错误，用法: temp 0.8")
                continue
            
            # 生成打字序列
            print("\n🔄 生成中...")
            sequence = model.generate(user_input, vocab, temperature=temperature, device=device)
            
            # 实时演示（会自动显示每一步的详细记录）
            animate_typing(sequence, speed_factor=speed)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python interact.py <模型文件.pth>")
        print("示例: python interact.py typing_model.pth")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("\n" + "="*70)
    print("🚀 加载模型...")
    print("="*70)
    
    try:
        model, vocab = load_model(model_path)
        print(f"✅ 模型加载成功!")
        print(f"📚 词汇表大小: {len(vocab)}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 设备: {device}")
        
        model = model.to(device)
        
        interactive_mode(model, vocab, device)
        
    except FileNotFoundError:
        print(f"❌ 找不到文件: {model_path}")
        print("\n请先训练模型:")
        print("  python train_typing_model.py")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
