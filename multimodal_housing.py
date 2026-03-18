import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. Dataset Definitions
# ==========================================

class MultimodalHousingDataset(Dataset):
    """
    Dataset for loading tabular data and images.
    If image_paths is None, it assumes the user is running the synthetic data generator.
    """
    def __init__(self, tabular_data, prices, image_dir=None, image_filenames=None, synthetic_images=None, transform=None):
        self.tabular_data = torch.tensor(tabular_data.values, dtype=torch.float32)
        self.prices = torch.tensor(prices.values, dtype=torch.float32).view(-1, 1)
        
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.synthetic_images = synthetic_images  # Used if no real images are provided
        self.transform = transform
        
    def __len__(self):
        return len(self.prices)
        
    def __getitem__(self, idx):
        # 1. Tabular features
        tab_features = self.tabular_data[idx]
        
        # 2. Image features
        if self.synthetic_images is not None:
            img = self.synthetic_images[idx]
        else:
            img_name = self.image_filenames.iloc[idx]
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
                
        # 3. Target value (Price)
        price = self.prices[idx]
        
        return tab_features, img, price

def generate_synthetic_data(num_samples=1000):
    """
    Generates dummy data to make the script immediately runnable.
    """
    print("No dataset provided via arguments. Generating synthetic dataset for demonstration...")
    np.random.seed(42)
    
    # Tabular features (e.g., bedrooms, bathrooms, area, age)
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    area = np.random.randint(500, 5000, num_samples)
    age = np.random.randint(0, 100, num_samples)
    
    # Target price: depends on features + noise
    price = 50 * area + 10000 * bedrooms + 10000 * bathrooms - 500 * age + np.random.normal(0, 50000, num_samples)
    
    df = pd.DataFrame({
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'area': area, 'age': age, 'price': price
    })
    
    # Synthetic images: Random noise (shape: C, H, W for PyTorch)
    # Using 3x64x64 for fast computation during demo
    images = np.random.rand(num_samples, 3, 64, 64).astype(np.float32)
    images = torch.tensor(images)
    
    return df, images

# ==========================================
# 2. Model Definitions
# ==========================================

class MultimodalModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalModel, self).__init__()
        
        # --- CNN Component for Images ---
        # Input shape expected: (Batch, 3, 64, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 32 x 16 x 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: 64 x 8 x 8
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )
        
        # --- MLP Component for Tabular Data ---
        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- Combined Output ---
        # Concatenate 128 (image feats) + 16 (tabular feats) = 144
        self.fc = nn.Sequential(
            nn.Linear(128 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Predict a single continuous value (Price) #
        )
        
    def forward(self, tab, img):
        # Extract features
        img_features = self.cnn(img)
        tab_features = self.mlp(tab)
        
        # Combine
        combined = torch.cat((img_features, tab_features), dim=1)
        
        # Predict
        out = self.fc(combined)
        return out

# ==========================================
# 3. Main Training & Evaluation Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Multimodal ML: Housing Price Prediction")
    parser.add_argument("--csv_path", type=str, help="Path to CSV containing tabular data and image filenames")
    parser.add_argument("--img_dir", type=str, help="Directory containing the house images")
    parser.add_argument("--img_col", type=str, default="image_name", help="Column name in CSV containing image filenames")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    synthetic_mode = False
    
    if args.csv_path and args.img_dir:
        # Load user-provided data
        print(f"Loading data from {args.csv_path} and images from {args.img_dir}")
        df = pd.read_csv(args.csv_path)
        
        # Ensure price column exists
        if 'price' not in df.columns:
            raise ValueError("The dataset CSV must contain a 'price' column as the prediction target.")
            
        tabular_cols = [col for col in df.columns if col not in ['price', args.img_col]]
        
        X_tabular = df[tabular_cols]
        X_images = df[args.img_col]
        y_prices = df['price']
        
        # Train-test split
        X_tab_train, X_tab_test, img_train_files, img_test_files, y_train, y_test = train_test_split(
            X_tabular, X_images, y_prices, test_size=0.2, random_state=42
        )
        
    else:
        # Generate synthetic data for demo matching assignment spec
        synthetic_mode = True
        df, images = generate_synthetic_data(1000)
        X_tabular = df.drop(columns=['price'])
        y_prices = df['price']
        
        # Train-test split
        X_tab_train, X_tab_test, img_train, img_test, y_train, y_test = train_test_split(
            X_tabular, images, y_prices, test_size=0.2, random_state=42
        )

    # Standardize tabular data
    scaler = StandardScaler()
    X_tab_train_scaled = pd.DataFrame(scaler.fit_transform(X_tab_train), columns=X_tab_train.columns)
    X_tab_test_scaled = pd.DataFrame(scaler.transform(X_tab_test), columns=X_tab_test.columns)
    
    # Image transformations (for real images)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Datasets
    if synthetic_mode:
        train_dataset = MultimodalHousingDataset(X_tab_train_scaled, y_train, synthetic_images=img_train)
        test_dataset = MultimodalHousingDataset(X_tab_test_scaled, y_test, synthetic_images=img_test)
    else:
        train_dataset = MultimodalHousingDataset(X_tab_train_scaled, y_train, image_dir=args.img_dir, image_filenames=img_train_files, transform=transform)
        test_dataset = MultimodalHousingDataset(X_tab_test_scaled, y_test, image_dir=args.img_dir, image_filenames=img_test_files, transform=transform)

    # Convert prices to thousands for numerically stable model training
    # Alternatively scale target, but we'll adapt model learning rate instead
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_tab_features = X_tab_train.shape[1]
    model = MultimodalModel(num_tabular_features=num_tab_features).to(device)
    
    criterion = nn.L1Loss() # Using MAE for loss to adapt better to large unscaled targets
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for tab, img, target in train_loader:
            tab, img, target = tab.to(device), img.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(tab, img)
            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * tab.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss (MAE): {epoch_loss:.2f}")

    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n--- Evaluating Model ---")
    with torch.no_grad():
        for tab, img, target in test_loader:
            tab, img, target = tab.to(device), img.to(device), target.to(device)
            outputs = model(tab, img)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    # Ensure arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate required metrics: MAE and RMSE
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(f"\nFinal Test Results:")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    
    # Save the trained model and scaler
    import pickle
    print("\nSaving the model and scaler to disk...")
    torch.save(model.state_dict(), "multimodal_model.pth")
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Model saved to 'multimodal_model.pth'.")
    print("Scaler saved to 'scaler.pkl'.")
    
    print("\nTraining and Evaluation Complete. Script terminated successfully.")

if __name__ == "__main__":
    main()
