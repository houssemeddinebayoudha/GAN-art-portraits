import streamlit as st
import torch
import random
import matplotlib.pyplot as plt
from utils import Generator, normalize_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_model = torch.load("Models/generator_315", weights_only=False, map_location=device)
st.set_page_config(
    page_title="Art Generator",
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded",
)

def generate_images(grid_cols):
    st.markdown("""
    <style>
    div.stSpinner > div {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>""", unsafe_allow_html=True)
    
    with st.spinner("Generating images..."):
        cols = st.columns(grid_cols)
        noise = torch.randn(num_images, 100, 1,1, device=device)
        fake_img = gen_model(noise).detach().cpu()
        for i in range(fake_img.size(0)):
            with cols[i % grid_cols]:
                st.image(normalize_tensor(fake_img[i]).permute(1, 2, 0).numpy(), use_column_width=True, clamp=True)

# Used to hide the image expand button
st.markdown(
    """
<style>
    [data-testid="StyledFullScreenButton"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
# Title of the app
st.title("Generating ART using a Simple GAN architecture 🎨")
# Brief description
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Generate a grid of AI-generated Art-like images using cutting-edge GAN models.\
             Simply select the number of images you'd like to see, and watch the magic happen!")
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.header("Generative adversarial networks")
st.sidebar.write("Generating ART using a Simple GAN architecture")
# Sidebar for configuration
st.sidebar.header("🛠️ Configuration")
# Input to choose the number of images
num_images = st.sidebar.slider("Select number of images", min_value=32, max_value=1028, value=128)
# Option to customize grid layout
grid_cols = st.sidebar.slider("Number of columns", min_value=8, max_value=20, value=10)
# Seed configuration
use_custom_seed = st.sidebar.checkbox("Provide custom seed?", value=False)

if use_custom_seed:
    # If user opts to provide a seed
    seed = st.sidebar.number_input("Enter your seed", value=0, step=1)
else:
    # Random seed
    seed = random.randint(0, 1000000)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.write(f"Seed being used: {seed}")

# Set the seed for PyTorch and other relevant libraries
torch.manual_seed(seed)
random.seed(seed)

generate_images(grid_cols)

st.markdown(
    """
    ---
    """
)
_, center, _ = st.columns([1,1,1])
# Footer section
center.markdown("""
👨‍💻 **Developed by Houssem Eddine Bayoudha**  
🔗 [GitHub Repository](https://github.com/houssemeddinebayoudha/GAN-art-portraits) | [LinkedIn](https://www.linkedin.com/in/houssem-bayoudha/)  
""")
