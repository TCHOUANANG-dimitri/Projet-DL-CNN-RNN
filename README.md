# Deep Learning Project: Image Classification and Time Series Forecasting

A comprehensive Deep learning project implementing advanced neural network architectures for two complementary tasks: image classification using Convolutional Neural Networks and time series forecasting using Long Short-Term Memory networks. The project includes complete training pipelines, evaluation metrics, and visualization tools for comprehensive model analysis.

## Project Overview

This repository presents a complete end-to-end solution for deep learning tasks using TensorFlow/Keras. The project demonstrates two distinct deep learning applications:

1. **CNN Image Classification**: Convolutional Neural Networks for CIFAR-10 object classification
2. **LSTM Time Series Forecasting**: Long Short-Term Memory networks for climatic data prediction on the Jena Climate dataset

Both components include full training pipelines, advanced regularization techniques, evaluation metrics, and comprehensive visualization tools.

## Datasets

### CIFAR-10 (Image Classification)

The CIFAR-10 dataset comprises 60,000 labeled 32x32 pixel RGB images across 10 object categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

**Dataset Statistics:**
- Training samples: 50,000
- Test samples: 10,000
- Image dimensions: 32x32 pixels (RGB)
- Normalization: Pixel values scaled to [0, 1]

### Jena Climate (Time Series Forecasting)

The Jena Climate dataset contains high-frequency climate measurements from the Max Planck Institute in Jena, Germany, spanning from 2009 to 2016.

**Dataset Features:**
- Temporal coverage: January 1, 2009 - December 31, 2016
- Measurement frequency: 10-minute intervals
- Number of records: ~420,000 time steps
- Variables tracked: 14 meteorological parameters

**Meteorological Variables:**
- p (mbar): Atmospheric pressure
- T (degC): Temperature
- Tpot (K): Potential temperature
- Tdew (degC): Dew point temperature
- rh (%): Relative humidity
- VPmax (mbar): Saturation vapor pressure
- VPact (mbar): Actual vapor pressure
- VPdef (mbar): Vapor pressure deficit
- sh (g/kg): Specific humidity
- H2OC (mmol/mol): Water vapor concentration
- rho (g/m³): Air density
- wv (m/s): Wind velocity
- max. wv (m/s): Maximum wind velocity
- wd (deg): Wind direction (degrees)

## Project Structure

```
├── Best_model.keras              # Serialized best-performing model
├── data/
│   └── jena_climate_2009_2016.csv
├── figures/                       # Generated visualizations and reports
│   ├── model_summary.txt
│   ├── training_history.json
│   ├── cnn/                       # CNN-specific visualizations
│   └── lstm/                      # LSTM-specific visualizations
├── models/
│   ├── __init__.py
│   ├── cnn_model.py               # Custom CNN architecture
│   └── rnn_model.py               # RNN architecture placeholder
└── utils/
    ├── __init__.py
    ├── data_loader.py             # Data loading and preprocessing
    ├── train.py                   # Training pipeline
    ├── evaluate.py                # Model evaluation and metrics
    └── visualization.py           # Visualization utilities
```

## Model Architectures

### CustomCNN (Image Classification)

The primary CNN model implements a multi-layer convolutional neural network with the following components:

**Convolutional Blocks:**
- Layer 1: 32 filters, 3x3 kernel with ReLU activation, L2 regularization
- Layer 2: 64 filters, 3x3 kernel with ReLU activation, L2 regularization
- Layer 3: 128 filters, 3x3 kernel with ReLU activation, L2 regularization

**Regularization Techniques:**
- Batch Normalization for stable training
- Dropout (rate: 0.4) for overfitting prevention
- L2 regularization (λ=1e-4) for weight constraint
- Max pooling layers for spatial dimension reduction

### LSTM (Time Series Forecasting)

The LSTM architecture is designed for sequential time series prediction on meteorological data. The model captures temporal dependencies and long-range patterns in climate measurements.

**Architecture Components:**
- Input layer: Temporal sequences of meteorological variables
- LSTM cells: Multi-unit LSTM layers capturing long-term dependencies
- Dropout layers: Regularization between LSTM layers
- Dense output layers: Regression heads for continuous value prediction
- Activation functions: ReLU for hidden layers, linear for output (regression)

**Key Features:**
- Sequential encoding of 10-minute interval climate data
- Multi-step ahead forecasting capability
- Capture of seasonal and diurnal climate patterns
- Handling of 14 interdependent meteorological features

## Data Processing

### Preprocessing Pipeline

The data loader implements the following preprocessing steps:

1. **Normalization**: Pixel values divided by 255 to scale to [0, 1] range
2. **Data Augmentation**: Applied during training to enhance model robustness
   - Random horizontal flipping
   - Random rotation (±10%)
   - Random translation (±10% height/width)
   - Random zoom (±10%)
   - Random contrast adjustment (±10%)
3. **Dataset Batching**: Batch size of 64 samples with automatic performance tuning

## Training Pipeline

### CNN Training

The CNN training process for CIFAR-10 incorporates advanced optimization techniques:

**Optimization Settings:**
- Optimizer: Adam with cosine decay learning rate scheduling
- Initial learning rate: 1e-3
- Decay steps: 50 epochs with batch-based scheduling
- Loss function: Sparse Categorical Cross-entropy
- Batch size: 64 samples

**Training Callbacks:**
- Early stopping with patience of 7 epochs and minimum delta of 0.001
- Model checkpointing to save best weights based on validation loss
- Learning rate reduction on plateau for adaptive training
- Training history logging for performance analysis

**Dataset Split:**
- Training: 50,000 samples
- Validation: Portion of training data for hyperparameter tuning
- Test: 10,000 samples for final evaluation

### LSTM Training

The LSTM training process for time series forecasting is configured for long-sequence predictions on climatic data:

**Optimization Configuration:**
- Optimizer: Adam with adaptive learning rate
- Loss function: Mean Squared Error (MSE) for regression tasks
- Metrics: MAE (Mean Absolute Error) for interpretable error measurement
- Batch size: Optimized for temporal sequence processing

**Training Strategy:**
- Sequential batching respecting temporal order
- Sliding window approach for multi-step prediction
- Normalization of all meteorological features
- Train/validation/test split preserving temporal integrity

## Evaluation and Metrics

### CNN Evaluation

The CNN evaluation module provides comprehensive image classification assessment:

**Performance Metrics:**
- Confusion matrix analysis for all 10 classes
- Per-class precision, recall, and F1-score
- Classification report with detailed statistics
- Prediction visualization and error analysis
- Misclassified sample identification

**Outputs:**
- Model architecture summary (text format)
- Training history (JSON format)
- Visualization plots for all metrics in `figures/cnn/` directory
- Detailed classification reports

### LSTM Evaluation

The LSTM evaluation module assesses time series forecasting performance:

**Forecasting Metrics:**
- Mean Absolute Error (MAE) for prediction accuracy
- Root Mean Squared Error (RMSE) for penalty on large errors
- Mean Absolute Percentage Error (MAPE) for relative performance
- Residual analysis and prediction confidence intervals
- Multi-step ahead forecast comparison with actual values

**Outputs:**
- Prediction vs. actual plots in `figures/lstm/` directory
- Residual distribution analysis
- Temporal pattern visualization
- Model convergence curves during training

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualizations)

### Setup

1. Clone or download the repository
2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

### CNN Image Classification

#### Training the CNN Model

Execute the CNN training pipeline:
```bash
python utils/train.py
```

This will:
- Load and preprocess CIFAR-10 data with augmentation
- Initialize the CustomCNN model
- Train for up to 50 epochs with early stopping
- Save the best model to `Best_model.keras`
- Log training history to `figures/training_history.json`

#### Evaluating the CNN Model

Generate comprehensive CNN evaluation metrics and visualizations:
```bash
python utils/evaluate.py
```

Or specify a custom model:
```bash
python utils/evaluate.py --model Best_model.keras
```

This will:
- Load the specified CNN model
- Evaluate performance on the CIFAR-10 test set
- Generate confusion matrices and classification reports
- Create visualization plots in `figures/cnn/` directory
- Save model architecture summary to `figures/model_summary.txt`

### LSTM Time Series Forecasting

#### Training the LSTM Model

To train the LSTM model on Jena Climate data (when implemented):
```bash
python models/rnn_model.py
```

Expected operations:
- Load and normalize Jena Climate time series data
- Create sequential batches preserving temporal order
- Train LSTM for multi-step climate variable prediction
- Save best performing model
- Log training history and metrics

#### Evaluating LSTM Predictions

To evaluate LSTM forecasting performance (when implemented):
```bash
python utils/evaluate.py --model lstm
```

Expected outputs:
- Prediction accuracy on held-out test sequences
- Multi-step ahead forecast visualization
- Residual analysis and error metrics
- Temporal pattern comparison plots in `figures/lstm/` directory

## File Descriptions

| File | Purpose |
|------|---------|
| `models/cnn_model.py` | CustomCNN class implementing the convolutional architecture |
| `utils/data_loader.py` | Data loading, preprocessing, and augmentation pipeline |
| `utils/train.py` | Complete training loop with callbacks and optimization |
| `utils/evaluate.py` | Model evaluation with metrics and visualization generation |
| `utils/visualization.py` | Plotting and visualization utilities |

## Model Performance

The best trained model is serialized and saved as `Best_model.keras`. Performance metrics and visualizations are generated during evaluation and stored in the `figures/` directory for analysis.

## Key Features

- Modular architecture supporting two distinct deep learning tasks
- Comprehensive CNN with advanced regularization for image classification
- LSTM framework for sequential time series prediction
- Complete data augmentation pipeline for CNN training
- Advanced regularization techniques (batch norm, dropout, L2 penalty)
- Adaptive learning rate scheduling for optimal convergence
- Detailed evaluation metrics and visualization tools for both tasks
- Model checkpointing and early stopping for efficient training
- Reproducible pipeline with configurable parameters
- Separate visualization directories for CNN and LSTM analyses

## Project Status

- **CNN Component**: Fully implemented and operational
- **LSTM Component**: Architecture template in place; ready for time series model implementation

The LSTM implementation (`models/rnn_model.py`) provides a foundation for sequence-to-sequence forecasting on the Jena Climate dataset. Training and evaluation functions support both architectures, with separate visualization pipelines for each.

## Future Enhancements

### CNN Improvements
- Implementation of advanced architectures (ResNet, EfficientNet, Vision Transformers)
- Ensemble methods combining multiple CNN models
- Transfer learning with ImageNet pretrained weights
- Model quantization for edge deployment optimization

### LSTM Improvements
- Implementation of complete LSTM training pipeline for Jena Climate data
- Exploration of attention mechanisms for temporal pattern recognition
- Bidirectional LSTM for improved context understanding
- Ensemble approaches combining LSTM with other time series models (Prophet, ARIMA)
- Multi-task learning predicting multiple meteorological variables simultaneously

### Cross-Task Enhancements
- Hyperparameter optimization framework using Bayesian search
- Automated model selection between architectures
- Comparative performance analysis across datasets
- Production deployment pipeline with containerization
- Real-time prediction inference capabilities

## License

This project is provided as-is for educational and research purposes.
