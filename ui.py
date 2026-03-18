import streamlit as st
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import pickle

# Import the model architecture from parallel training script
from multimodal_housing import MultimodalModel

st.set_page_config(page_title="Multimodal Housing Predictor", page_icon="🏠", layout="centered")

# Custom CSS for better UI look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2e3b4e;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    scaler = None
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("`scaler.pkl` not found! Please run `python multimodal_housing.py` first to train and save the model.")
        
    model = None
    if scaler is not None:
        # We manually know the dummy data features count is 4 for demonstration
        model = MultimodalModel(num_tabular_features=4) 
        try:
            model.load_state_dict(torch.load('multimodal_model.pth', map_location=torch.device('cpu')))
            model.eval()
        except FileNotFoundError:
            st.error("`multimodal_model.pth` not found! Please run `python multimodal_housing.py` first to train and save the model.")
            model = None
            
    return scaler, model

def main():
    st.title("🏡 Multimodal Housing Price Predictor")
    st.markdown("""
    This application predicts housing prices using a **Multimodal Machine Learning Model** (PyTorch). 
    It combines **Tabular Features** (Standardized via StandardScaler) processed by an MLP and an **Image** processed by a CNN.
    """)
    
    scaler, model = load_assets()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Tabular Features")
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, step=100, value=1500)
        age = st.number_input("Age of Home (years)", min_value=0, max_value=200, value=10)

    with col2:
        st.subheader("🖼️ House Image")
        uploaded_file = st.file_uploader("Upload an image of the house...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Show a placeholder image for aesthetics if none uploaded yet
        else:
            st.info("Please upload an image to see the prediction.")

    st.markdown("---")
    
    if st.button("🔮 Predict Price"):
        if model is None or scaler is None:
            st.warning("Cannot predict: Model or Scaler assets are missing.")
        elif uploaded_file is None:
            st.warning("Please upload an image of the house first!")
        else:
            with st.spinner("Analyzing image and features..."):
                # 1. Process Tabular Data
                tabular_data = pd.DataFrame([{
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'area': area,
                    'age': age
                }])
                
                tabular_scaled = scaler.transform(tabular_data)
                tab_tensor = torch.tensor(tabular_scaled, dtype=torch.float32)
                
                # 2. Process Image Data
                # (Matches the training script preprocessing for synthetic dummy dataset)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()
                ])
                img_tensor = transform(image).unsqueeze(0) # Add batch dimension [1, C, H, W]
                
                # 3. Run Inference
                with torch.no_grad():
                    prediction = model(tab_tensor, img_tensor)
                    price_usd = prediction.item()
                    price_pkr = price_usd * 278.0 # Approximate conversion to PKR
                    
                st.success(f"## Predicted Value: **PKR {price_pkr:,.2f}**")
                if price_usd < 0:
                    st.caption("*(Note: Negative prices can occur with randomly initialized demo models due to noise variance)*")

if __name__ == "__main__":
    main()
