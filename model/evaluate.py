import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from pathlib import Path
import pandas as pd
from data_processor import DataProcessor
from model import HEAModel

class ModelEvaluator:
    def __init__(self, model, data_processor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.data_processor = data_processor
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Create output directory for plots
        self.output_dir = Path('evaluation_results')
        self.output_dir.mkdir(exist_ok=True)
        
    def evaluate_model(self, test_loader):
        """Evaluate model on test data and return metrics"""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }
        
        return metrics, predictions, targets
    
    def plot_predictions_vs_targets(self, predictions, targets, phase_names):
        """Plot predictions vs targets for each phase"""
        n_phases = len(phase_names)
        fig, axes = plt.subplots(n_phases, 1, figsize=(10, 5*n_phases))
        
        for i, (ax, phase_name) in enumerate(zip(axes, phase_names)):
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.5)
            ax.plot([targets[:, i].min(), targets[:, i].max()], 
                   [targets[:, i].min(), targets[:, i].max()], 
                   'r--', label='Perfect Prediction')
            
            r2 = r2_score(targets[:, i], predictions[:, i])
            ax.set_title(f'{phase_name} (R² = {r2:.3f})')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions_vs_targets.png')
        plt.close()
    
    def plot_error_distribution(self, predictions, targets):
        """Plot error distribution"""
        errors = predictions - targets
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors.flatten(), kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'error_distribution.png')
        plt.close()
    
    def plot_metrics_by_phase(self, predictions, targets, phase_names):
        """Plot metrics for each phase"""
        metrics_by_phase = []
        
        for i, phase_name in enumerate(phase_names):
            phase_metrics = {
                'Phase': phase_name,
                'MSE': mean_squared_error(targets[:, i], predictions[:, i]),
                'RMSE': np.sqrt(mean_squared_error(targets[:, i], predictions[:, i])),
                'MAE': mean_absolute_error(targets[:, i], predictions[:, i]),
                'R²': r2_score(targets[:, i], predictions[:, i])
            }
            metrics_by_phase.append(phase_metrics)
        
        metrics_df = pd.DataFrame(metrics_by_phase)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_df.plot(x='Phase', y='MSE', kind='bar', ax=axes[0,0])
        axes[0,0].set_title('MSE by Phase')
        
        metrics_df.plot(x='Phase', y='RMSE', kind='bar', ax=axes[0,1])
        axes[0,1].set_title('RMSE by Phase')
        
        metrics_df.plot(x='Phase', y='MAE', kind='bar', ax=axes[1,0])
        axes[1,0].set_title('MAE by Phase')
        
        metrics_df.plot(x='Phase', y='R²', kind='bar', ax=axes[1,1])
        axes[1,1].set_title('R² by Phase')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_by_phase.png')
        plt.close()
        
        # Save metrics to CSV
        metrics_df.to_csv(self.output_dir / 'metrics_by_phase.csv', index=False)
    
    def save_evaluation_results(self, metrics):
        """Save evaluation metrics to a text file"""
        with open(self.output_dir / 'evaluation_metrics.txt', 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("======================\n\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")

def main():
    # Load data and model
    data_processor = DataProcessor('../datasets/data.csv')
    _, _, test_loader = data_processor.prepare_dataloaders()
    
    # Load the trained model
    model = HEAModel(
        input_size=len(data_processor.get_feature_names()),
        output_size=data_processor.get_output_size()
    )
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Create evaluator
    evaluator = ModelEvaluator(model, data_processor)
    
    # Evaluate model
    metrics, predictions, targets = evaluator.evaluate_model(test_loader)
    
    # Get phase names
    phase_names = data_processor.get_target_names()
    
    # Generate plots and save results
    evaluator.plot_predictions_vs_targets(predictions, targets, phase_names)
    evaluator.plot_error_distribution(predictions, targets)
    evaluator.plot_metrics_by_phase(predictions, targets, phase_names)
    evaluator.save_evaluation_results(metrics)
    
    print("Evaluation completed. Results saved in 'evaluation_results' directory.")

if __name__ == "__main__":
    main() 