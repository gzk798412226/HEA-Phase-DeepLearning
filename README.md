# HEA Phase DeepLearning

This repository contains a deep learning model for predicting phases in High Entropy Alloys (HEAs) based on their composition and properties.

## Overview

High Entropy Alloys (HEAs) are a class of materials that contain multiple principal elements in equal or near-equal atomic percentages. Predicting the phases in HEAs is crucial for material design and optimization. This project implements a deep learning model to predict multiple phase categories in HEAs based on their elemental composition and physical properties.

## Features

- Multi-output regression model for phase prediction
- Handles multiple phase categories simultaneously
- Comprehensive data preprocessing pipeline
- Detailed model evaluation and visualization tools
- Support for both training and inference

## Model Performance

The model achieves the following performance metrics:

- Overall R² Score: 0.505
- RMSE: 0.1789
- MAE: 0.0818

Performance by phase category:
- Category-3: R² = 0.943 (Excellent)
- Category-2: R² = 0.094 (Moderate)
- Category-1 & Phases: R² ≈ 0 (Needs improvement)
- Category-4: R² = 0.037 (Needs improvement)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HEA-Phase-DeepLearning.git
cd HEA-Phase-DeepLearning
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
HEA-phase-prediction-database/
├── datasets/
│   └── data.csv
├── model/
│   ├── data_processor.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── evaluation_results/
│   ├── evaluation_metrics.txt
│   ├── metrics_by_phase.csv
│   └── various_plots.png
├── requirements.txt
└── README.md
```

## Usage

### Training the Model

```bash
python model/train.py
```

### Evaluating the Model

```bash
python model/evaluate.py
```

### Making Predictions

```python
from model.data_processor import DataProcessor
from model.model import HEAModel

# Load and preprocess data
data_processor = DataProcessor('path_to_data.csv')
X = data_processor.preprocess_data(your_data)

# Load trained model
model = HEAModel(input_size=X.shape[1], output_size=number_of_phases)
model.load_state_dict(torch.load('best_model.pth'))

# Make predictions
predictions = model(X)
```

## Model Architecture

The model uses a deep neural network with the following features:
- Multiple hidden layers with batch normalization
- ReLU activation functions
- Dropout for regularization
- Adam optimizer
- MSE loss function

## Data

The dataset includes:
- Elemental compositions
- Physical properties
- Phase information for multiple categories

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{HEA_phase_prediction,
  author = {Zikun Guo},
  title = {HEA Phase DeepLearning},
  year = {2024},
  url = {https://github.com/yourusername/HEA-Phase-DeepLearning}
}
```

## Contact

For any questions or suggestions, please open an issue in the GitHub repository. Or gzk798412226@gmail.com