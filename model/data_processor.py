import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class HEADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.element_columns = None
        self.property_columns = None
        self.phase_columns = None
        self.output_size = None  # Will store the size after one-hot encoding
        
    def load_data(self):
        """Load and preprocess the data"""
        df = pd.read_csv(self.data_path)
        
        # Identify element columns (they are named with element symbols)
        self.element_columns = [col for col in df.columns if col in ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'V', 'Ti', 'Mo', 'Si', 'Y', 'Nb', 'Nd', 'Zr', 'Ge', 'Sn', 'Ta', 'Hf', 'W', 'Mg', 'Li', 'Zn', 'Dy', 'Tb', 'Tm', 'B', 'C', 'Be', 'Ce', 'La', 'Er', 'Pr']]
        
        # Identify property columns (they start with D)
        self.property_columns = [col for col in df.columns if col.startswith('D')]
        
        # Identify phase columns
        self.phase_columns = ['Phases', 'category-1', 'category-2', 'category-3', 'category-4']
        
        # Prepare features (X) - element compositions and properties
        X = df[self.element_columns + self.property_columns]
        
        # Prepare targets (y) - phase information
        y = df[self.phase_columns]
        
        return X, y
    
    def prepare_dataloaders(self, batch_size=32, test_size=0.2, val_size=0.1):
        """Prepare PyTorch DataLoaders for training, validation, and testing"""
        # Load and preprocess the data
        X, y = self.load_data()
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode the targets
        y_encoded = self.ohe.fit_transform(y)
        self.output_size = y_encoded.shape[1]  # Store the output size after encoding
        
        # Create datasets
        train_dataset = HEADataset(X_scaled, y_encoded)
        
        # Calculate split sizes
        total_size = len(train_dataset)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)
        train_size = total_size - test_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def get_feature_names(self):
        """Get the names of features used in the model"""
        return self.element_columns + self.property_columns
    
    def get_target_names(self):
        """Get the names of target variables"""
        return self.phase_columns
    
    def get_output_size(self):
        """Get the size of the output after one-hot encoding"""
        if self.output_size is None:
            # If not already computed, compute it
            _, y = self.load_data()
            y_encoded = self.ohe.fit_transform(y)
            self.output_size = y_encoded.shape[1]
        return self.output_size
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """Split data into train, validation and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation set from remaining data
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def inverse_transform_properties(self, scaled_properties):
        """Convert scaled properties back to original scale"""
        return self.scaler.inverse_transform(scaled_properties)
    
    def preprocess_data(self, X, y):
        # 对输入特征进行标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 对目标变量进行标准化
        y_scaled = self.scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def inverse_transform_y(self, y_scaled):
        """将标准化的目标变量转换回原始尺度"""
        return self.scaler.inverse_transform(y_scaled) 