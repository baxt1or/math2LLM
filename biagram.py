import torch 
import torch.nn as nn
from torch.nn import functional as F


file_path = "/Users/baxtiyorbekmurodov/Desktop/math2LLM/data/ikki_eshik_orasi.txt"

batch_size = 32
block_size = 8
epochs = 10000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 32



torch.manual_seed(1337)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = {c:i for i,c in enumerate(chars)}
itoi = {i:c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda tokens: ''.join([itoi[t] for t in tokens])


data = torch.tensor(encode(text), dtype=torch.long)


n = len(data)
tr_size = int(n * 0.9)
train_data = data[:tr_size]
val_data = data[tr_size:]


def get_batch(split :str):
    # generates small batch for X input and y target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y




@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train','val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out




class BiagramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)  
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    


model = BiagramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for i in range(epochs):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'step: {i}, train loss={losses["train"]:.4f} and val loss={losses["val"]:.4f}')

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(idx, 700)[0].tolist()))