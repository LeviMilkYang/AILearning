import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from PositionalEncoding import PositionalEncoding
from SimpleTokenizer import SimpleTokenizer
from TransformerBlock import TransformerBlock


DEMO_CORPUS = "hello transformer, lets build a gpt step by step!"


class CharacterGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, context_length):
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=context_length)
        self.tf_block = TransformerBlock(d_model, num_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        x = self.pos_encoder(self.embedding(token_ids))
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        return self.lm_head(self.tf_block(x, mask=mask))


def train(corpus, steps, checkpoint_path):
    torch.manual_seed(42)
    tokenizer = SimpleTokenizer(corpus)
    config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 64,
        "num_heads": 4,
        "context_length": min(16, len(corpus) - 1),
    }
    model = CharacterGPT(**config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    token_ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    context_length = config["context_length"]
    offsets = torch.arange(context_length)

    for step in range(1, steps + 1):
        # 每个窗口的目标整体右移一位，让每个位置预测下一个字符。
        starts = torch.randint(len(token_ids) - context_length, (16, 1))
        positions = starts + offsets
        inputs = token_ids[positions]
        targets = token_ids[positions + 1]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == steps:
            print(f"step {step}/{steps}  loss={loss.item():.4f}")

    torch.save(
        {
            "config": config,
            # SimpleTokenizer 按字符排序分配 ID，因此保存词表即可重建映射。
            "vocab": tokenizer.decode(range(tokenizer.vocab_size)),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


@torch.inference_mode()
def generate(checkpoint_path, prompt, max_new_tokens):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = SimpleTokenizer(checkpoint["vocab"])
    model = CharacterGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    token_ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        # 只用最近的上下文预测下一字符，同时保留完整文本用于输出。
        context = torch.tensor([token_ids[-model.context_length:]], dtype=torch.long)
        logits = model(context)
        next_id = logits[0, -1].argmax().item()
        token_ids.append(next_id)

    return tokenizer.decode(token_ids)


def main():
    parser = argparse.ArgumentParser(description="Train or run a character-level GPT.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--corpus-file", type=Path, help="Optional UTF-8 training text.")
    train_parser.add_argument("--steps", type=int, default=200)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("model.pt"))

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=30)
    generate_parser.add_argument("--checkpoint", type=Path, default=Path("model.pt"))

    args = parser.parse_args()
    # 小模型在 CPU 上用单线程，减少线程调度开销。
    torch.set_num_threads(1)
    if args.command == "train":
        corpus = args.corpus_file.read_text(encoding="utf-8") if args.corpus_file else DEMO_CORPUS
        if len(corpus) < 2:
            train_parser.error("the corpus must contain at least two characters")
        if args.steps < 1:
            train_parser.error("--steps must be positive")
        train(corpus, args.steps, args.checkpoint)
    else:
        if not args.prompt:
            generate_parser.error("--prompt must not be empty")
        if args.max_new_tokens < 0:
            generate_parser.error("--max-new-tokens must be non-negative")
        try:
            print(generate(args.checkpoint, args.prompt, args.max_new_tokens))
        except KeyError as error:
            generate_parser.error(f"character not in the training vocabulary: {error}")


if __name__ == "__main__":
    main()
