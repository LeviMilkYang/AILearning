class SimpleTokenizer:
    def __init__(self, text):
        # 1. 找出文本中所有不重复的字符，作为我们的词表
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # 2. 建立 字符 -> 数字ID 的映射
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # 3. 建立 数字ID -> 字符 的映射
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, s):
        # 把字符串转成数字列表
        return [self.stoi[c] for c in s]
        
    def decode(self, l):
        # 把数字列表转回字符串
        return ''.join([self.itos[i] for i in l])

# --- 验证一下 ---
if __name__ == "__main__":
    corpus = "hello transformer, lets build a gpt!"
    tokenizer = SimpleTokenizer(corpus)
    
    encoded_text = tokenizer.encode("hello")
    print("文字转数字 ID:", encoded_text)
    
    decoded_text = tokenizer.decode(encoded_text)
    print("数字 ID 转回文字:", decoded_text)