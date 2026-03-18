# Multimodal Housing Price Predictor

This repository contains a PyTorch multimodal machine learning model for predicting housing prices using both tabular features and house images. It includes a frontend Streamlit web application.

## Files Structure

- `multimodal_housing.py`: The main pipeline script that defines the model architecture, generates a synthetic dataset, and trains the model.
- `ui.py`: The user-friendly Streamlit web interface.
- `multimodal_model.pth`: The trained model weights.
- `scaler.pkl`: The saved StandardScaler mapping for normalizing tabular inputs.
- `requirements.txt`: Python dependencies required to run the UI.

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit UI:
   ```bash
   streamlit run ui.py
   ```

*(Optionally, you can retrain the model by running `python multimodal_housing.py`)*
