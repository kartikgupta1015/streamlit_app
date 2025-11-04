# =====================================================
# 🧠 Next K Word Generator — Single File App
# =====================================================
# Author: Vandiita 🩵
# Powered by PyTorch + Streamlit
# =====================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np

# =========================
# Utility & Tokenizer
# =========================
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    word2idx = tokenizer.word_index
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word

stoi, itos = load_tokenizer()
vocab_size = len(stoi) + 1
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


# =========================
# Streamlit App
# =========================
set_seed(42)

st.set_page_config(page_title="🧠 Next K Word Generator", layout="centered")
st.title("🧩 Next K Word Generator")
st.caption("Generate the next K words using pretrained MLP models built in PyTorch.")
st.markdown("---")

# Model Variants
model_variants = {
    "Embedding 32 + ReLU": ("model_embed32_relu.pth", 32, "relu"),
    "Embedding 32 + Tanh": ("model_embed32_tanh.pth", 32, "tanh"),
    "Embedding 64 + ReLU": ("model_embed64_relu.pth", 64, "relu"),
    "Embedding 64 + Tanh": ("model_embed64_tanh.pth", 64, "tanh"),
}

# Sidebar Options
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select a Model Variant", list(model_variants.keys()))
context_len = st.sidebar.slider("Context Length (Window Size)", 3, 10, 5)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.5, 2.0, 1.0, 0.1)
num_words = st.sidebar.slider("Number of Words to Generate", 1, 30, 10)
st.sidebar.info("Random Seed fixed at **42** for reproducibility 🌱")

# Load Model
@st.cache_resource
def load_model(model_path, embed_dim, act_func, window_size):
    model = Next_Word_Predictor(
        vocab_size=len(stoi) + 1,
        embed_dim=embed_dim,
        activation=act_func,
        window_size=window_size
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model_state = model.state_dict()

    # Auto-fix mismatched keys
    for name, param in state_dict.items():
        if name in model_state and model_state[name].shape != param.shape:
            st.warning(f"Skipping layer {name} due to size mismatch.")
            continue
        elif name not in model_state:
            st.warning(f"Ignoring unexpected key: {name}")

    model_state.update({k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape})
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


path, embed_dim, act_func = model_variants[model_choice]
model = load_model(path, embed_dim, act_func, context_len)

# Text Input
st.subheader("✏️ Enter Your Text")
user_input = st.text_input("Your starting phrase:", "The world is full of")
generate_btn = st.button("✨ Generate Text")

if generate_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to begin.")
    else:
        words = user_input.strip().lower().split()
        unknown_words = [w for w in words if w not in stoi]
        if unknown_words:
            st.markdown("**Generated Text:**")
            st.markdown(f"> {user_input}")
        else:
            with st.spinner("Generating next words..."):
                result = generate_next_words(model, user_input, num_words, temperature, window_size=context_len)
                st.success("✅ Prediction Complete!")
                st.markdown("**Generated Text:**")
                st.markdown(f"> {result}")

st.markdown("---")
st.caption("Developed by Vandiita 🩵 | Fixed Random Seed = 42 | Powered by PyTorch ⚡")
