import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np

# =========================
# Load tokenizer and mappings
# =========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}
stoi = word2idx
itos = idx2word

vocab_size = len(word2idx) + 1
DEFAULT_WINDOW = 5  # default context length used during training


# =========================
# Model Definition
# =========================
class Next_Word_Predictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=1024, activation="tanh", dropout=0.4, window_size=DEFAULT_WINDOW):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * window_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.activation = activation
        self.window_size = window_size

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        else:
            x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# =========================
# Text Generation Function
# =========================
def generate_next_words(model, seed_text, next_words=10, temperature=1.0, window_size=DEFAULT_WINDOW):
    model.eval()
    words = seed_text.lower().split()

    for _ in range(next_words):
        # Convert to indices, pad or trim
        encoded = [stoi.get(w, 0) for w in words[-window_size:]]
        if len(encoded) < window_size:
            encoded = [0] * (window_size - len(encoded)) + encoded

        x = torch.tensor([encoded])

        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

        next_word = itos.get(next_idx, "<UNK>")
        words.append(next_word)

    return " ".join(words)


# =========================
# Random Seed (Fixed)
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
