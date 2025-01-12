import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# Hyperparameters
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 256
batch_size = 64
learning_rate = 3e-4
num_steps = 5000
embedding_dim = 384
num_layers = 6
num_heads = 6
dropout_rate = 0.2

# Load and preprocess data
with open("input.txt", "r") as file:
    text = file.read()

# Create vocabulary and mapping
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    """Encode a string into a list of integers."""
    return [stoi[c] for c in text]


def decode(indices):
    """Decode a list of integers back into a string."""
    return "".join(itos[i] for i in indices)


data = torch.tensor(encode(text), dtype=torch.int64)

# Split data into training and testing sets
split_idx = int(0.9 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]


def get_batch(data, batch_size):
    """Generate input-output batches for training/testing."""
    indices = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    return x.to(device), y.to(device)


# Define model components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) / (C**0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.attn = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(embedding_dim),
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(idx.size(1), device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        logits = logits.transpose(1, 2)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, :, -1]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Initialize model, optimizer, and loss function
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    """Estimate training and testing loss."""
    losses = {"train": 0, "test": 0}
    model.eval()
    for split, data in zip(["train", "test"], [train_data, test_data]):
        batch_losses = []
        for _ in range(100):  # Evaluate over 100 batches
            X, Y = get_batch(data, batch_size)
            _, loss = model(X, Y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


# Training loop with progress bar
with tqdm(total=num_steps, desc="Training Progress", leave=True) as pbar:
    for step in range(num_steps):
        xb, yb = get_batch(train_data, batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress every 100 steps
        if step % 100 == 0:
            losses = estimate_loss()
            tqdm.write(
                f"Step {step}: Train Loss {losses['train']:.4f}, Test Loss {losses['test']:.4f}"
            )
        pbar.update(1)

# Generate and print sample text
start_token = torch.zeros((1, 1), dtype=torch.int64, device=device)
generated_text = decode(model.generate(start_token, max_new_tokens=500)[0].cpu().tolist())
print(generated_text)
