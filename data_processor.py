import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.element_columns = None
        self.property_columns = None
        self.phase_columns = None
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the data from CSV file"""
        self.data = pd.read_csv(self.data_path)
        
        # Identify element columns (they are named with element symbols)
        self.element_columns = [col for col in self.data.columns if col in ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'V', 'Ti', 'Mo', 'Si', 'Y', 'Nb', 'Nd', 'Zr', 'Ge', 'Sn', 'Ta', 'Hf', 'W', 'Mg', 'Li', 'Zn', 'Dy', 'Tb', 'Tm', 'B', 'C', 'Be', 'Ce', 'La', 'Er', 'Pr']]
        
        # Identify property columns (they start with D)
        self.property_columns = [col for col in self.data.columns if col.startswith('D')]
        
        # Identify phase columns
        self.phase_columns = ['Phases', 'category-1', 'category-2', 'category-3', 'category-4']
        
        # Prepare features (X) - element compositions and properties
        X = self.data[self.element_columns + self.property_columns]
        
        # Prepare targets (y) - phase information
        y = self.data[self.phase_columns]
        
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
        y_encoded = self.encoder.fit_transform(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_encoded)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)
        train_size = total_size - test_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
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