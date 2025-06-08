import os
import torch
import numpy as np
from data_processor import DataProcessor
from model import HEAModel, HEATrainer
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data processor
    data_processor = DataProcessor('../datasets/data.csv')
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = data_processor.prepare_dataloaders(
        batch_size=32,
        test_size=0.2,
        val_size=0.1
    )
    
    # Get input and output sizes from the data
    input_size = len(data_processor.get_feature_names())
    output_size = data_processor.get_output_size()  # Get the size after one-hot encoding
    
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    
    # Create model
    model = HEAModel(
        input_size=input_size,
        hidden_sizes=[128, 256, 128],
        output_size=output_size,  # Use the actual number of target features after encoding
        dropout_rate=0.2
    )
    
    # Create trainer
    trainer = HEATrainer(
        model,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Train model
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,
        patience=10
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Evaluate final model
    final_val_loss = trainer.validate(val_loader)
    print(f'\nFinal Validation Loss: {final_val_loss:.4f}')

if __name__ == '__main__':
    main() 