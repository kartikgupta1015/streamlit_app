import streamlit as st
import torch
from model_utils import Next_Word_Predictor, generate_next_words, stoi, itos, set_seed

# Set fixed random seed
set_seed(42)

# ==========================
# Streamlit App Configuration
# ==========================
st.set_page_config(page_title="🧠 Next K Word Generator", layout="centered")
st.title("🧩 Next K Word Generator")
st.caption("Generate the next K words using pretrained MLP models built in PyTorch.")

st.markdown("---")

# ==========================
# Model Variants
# ==========================
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

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model(model_path, embed_dim, act_func, window_size):
    model = Next_Word_Predictor(
        vocab_size=len(stoi) + 1,
        embed_dim=embed_dim,
        activation=act_func,
        window_size=window_size
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # --- Auto-fix vocabulary size mismatches ---
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            if model_state[name].shape != param.shape:
                st.warning(f"Skipping layer {name} due to size mismatch.")
                continue
        else:
            st.warning(f"Ignoring unexpected key: {name}")
    model_state.update({k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape})
    model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


path, embed_dim, act_func = model_variants[model_choice]
model = load_model(path, embed_dim, act_func, context_len)

# ==========================
# Text Input Section
# ==========================
st.subheader("✏️ Enter Your Text")
user_input = st.text_input("Your starting phrase:", "The world is full of")
generate_btn = st.button("✨ Generate Text")

if generate_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to begin.")
    else:
        words = user_input.strip().lower().split()

        # ✅ Check if all input words are in the dataset (vocab)
        unknown_words = [w for w in words if w not in stoi]
        if unknown_words:
            # 🧩 Just display the input text unchanged
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