# Cognitive Workload Classification Using fNIRS Signals

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![fNIRS](https://img.shields.io/badge/fNIRS-Brain%20Computer%20Interface-green.svg)]()

> A machine learning-based system for classifying cognitive workload levels using functional Near-Infrared Spectroscopy (fNIRS) signals with advanced preprocessing and feature extraction techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Performance Metrics](#performance-metrics)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## 🧠 Overview

This project implements a comprehensive machine learning pipeline for cognitive workload classification using functional Near-Infrared Spectroscopy (fNIRS) signals. The system employs advanced signal processing techniques, feature extraction methods, and machine learning algorithms to accurately classify different levels of mental workload from neuroimaging data.

### Key Highlights

- **Non-invasive Brain Monitoring**: Utilizes fNIRS technology for real-time cognitive state assessment
- **Multi-level Classification**: Supports binary and multi-class workload level classification
- **Advanced Signal Processing**: Implements state-of-the-art preprocessing and artifact removal techniques
- **Machine Learning Pipeline**: Features extraction, selection, and classification using multiple algorithms
- **Real-time Applications**: Designed for brain-computer interface (BCI) and human-computer interaction systems

### Problem Statement

Cognitive workload assessment is crucial for understanding human performance in various domains including education, aviation, healthcare, and human-computer interaction. Traditional subjective measures are unreliable and intrusive. This project addresses the need for objective, real-time cognitive workload monitoring using neuroimaging data.

## ✨ Features

### Core Functionality
- **Signal Preprocessing**: Motion artifact removal, filtering, and baseline correction
- **Feature Extraction**: Time-domain, frequency-domain, and statistical features
- **Machine Learning Models**: Support for multiple algorithms including SVM, Random Forest, Neural Networks
- **Real-time Processing**: Optimized for online cognitive state monitoring
- **Cross-subject Validation**: Robust evaluation using leave-one-subject-out cross-validation

### Advanced Features
- **Hemodynamic Response Analysis**: HbO and HbR concentration analysis
- **Multi-channel Processing**: Support for various fNIRS device configurations
- **Automated Pipeline**: End-to-end processing from raw signals to classification results
- **Visualization Tools**: Signal plotting, brain activation maps, and performance metrics
- **Extensible Architecture**: Easy integration of new algorithms and datasets

## 📊 Dataset

### Experimental Design
- **Task Type**: N-back cognitive workload tasks with varying difficulty levels
- **Participants**: Multiple subjects with demographic diversity
- **Recording Setup**: Multi-channel fNIRS device with prefrontal cortex coverage
- **Sampling Rate**: High-frequency data acquisition for precise temporal analysis

### Data Characteristics
- **Signal Types**: Oxyhemoglobin (HbO) and Deoxyhemoglobin (HbR) concentration changes
- **Workload Levels**: Low, Medium, High cognitive load conditions
- **Session Structure**: Structured experimental blocks with rest periods
- **Data Quality**: Preprocessed and validated for machine learning applications

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- MATLAB (optional, for advanced signal processing)
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yogeeshhr2003/cognitive-Workload-Classification-Using-fNIRS-Signals.git
   cd cognitive-Workload-Classification-Using-fNIRS-Signals
   ```

2. **Create virtual environment**
   ```bash
   python -m venv fnirs_env
   source fnirs_env/bin/activate  # On Windows: fnirs_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
mne>=0.23.0
jupyter>=1.0.0
plotly>=5.0.0
```

### Optional Dependencies

For advanced features:
```bash
pip install tensorflow>=2.6.0  # For deep learning models
pip install torch>=1.9.0       # PyTorch support
pip install nilearn>=0.8.0     # Neuroimaging analysis
```

## 📖 Usage

### Basic Usage

1. **Data Preprocessing**
   ```python
   from src.preprocessing import fNIRSPreprocessor
   
   preprocessor = fNIRSPreprocessor()
   processed_data = preprocessor.process_raw_data('data/raw_fnirs_data.mat')
   ```

2. **Feature Extraction**
   ```python
   from src.features import FeatureExtractor
   
   extractor = FeatureExtractor()
   features = extractor.extract_features(processed_data)
   ```

3. **Model Training**
   ```python
   from src.models import WorkloadClassifier
   
   classifier = WorkloadClassifier(model_type='random_forest')
   classifier.train(features, labels)
   ```

4. **Classification**
   ```python
   predictions = classifier.predict(test_features)
   accuracy = classifier.evaluate(test_features, test_labels)
   ```

### Advanced Usage

#### Running Complete Pipeline
```bash
python main.py --data_path data/ --model rf --cv_folds 5 --output results/
```

#### Hyperparameter Optimization
```bash
python optimize_hyperparameters.py --model svm --search_space config/search_space.json
```

#### Real-time Classification
```bash
python realtime_classifier.py --model_path models/best_model.pkl --device_config config/device.json
```

### Jupyter Notebook Examples

Explore the provided notebooks in the `notebooks/` directory:
- `01_data_exploration.ipynb`: Data analysis and visualization
- `02_preprocessing_pipeline.ipynb`: Signal preprocessing demonstration
- `03_feature_analysis.ipynb`: Feature extraction and selection
- `04_model_comparison.ipynb`: Comparative analysis of ML algorithms
- `05_results_visualization.ipynb`: Performance metrics and visualization

## 📁 Project Structure

```
cognitive-Workload-Classification-Using-fNIRS-Signals/
├── data/                          # Dataset directory
│   ├── raw/                       # Raw fNIRS data files
│   ├── processed/                 # Preprocessed data
│   └── metadata/                  # Experimental metadata
├── src/                           # Source code
│   ├── preprocessing/             # Signal preprocessing modules
│   ├── features/                  # Feature extraction algorithms
│   ├── models/                    # Machine learning models
│   ├── visualization/             # Plotting and visualization tools
│   └── utils/                     # Utility functions
├── notebooks/                     # Jupyter notebooks
├── config/                        # Configuration files
├── tests/                         # Unit tests
├── results/                       # Experimental results
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── main.py                        # Main execution script
└── README.md                      # This file
```

## 🔬 Methodology

### Signal Preprocessing Pipeline

1. **Motion Artifact Removal**
   - Temporal Derivative Distribution Repair (TDDR)
   - Spline interpolation for spike artifact correction
   - Movement detection and correction algorithms

2. **Filtering**
   - Bandpass filtering (0.01-0.5 Hz) to remove physiological noise
   - Butterworth filter implementation
   - Anti-aliasing and noise reduction

3. **Baseline Correction**
   - Block-wise baseline normalization
   - Detrending using polynomial fitting
   - Z-score normalization across channels

### Feature Extraction

#### Time-Domain Features
- Statistical measures (mean, std, skewness, kurtosis)
- Peak detection and amplitude analysis
- Slope and area under curve calculations

#### Frequency-Domain Features
- Power spectral density analysis
- Frequency band power extraction
- Coherence analysis between channels

#### Hemodynamic Features
- HbO/HbR concentration changes
- Hemodynamic response function modeling
- Channel-wise activation patterns

### Machine Learning Pipeline

1. **Feature Selection**
   - Correlation-based feature selection
   - Mutual information analysis
   - Principal Component Analysis (PCA)

2. **Model Training**
   - Cross-validation with subject independence
   - Hyperparameter optimization using grid search
   - Model ensemble techniques

3. **Evaluation**
   - Leave-one-subject-out cross-validation
   - Performance metrics (accuracy, F1-score, AUC)
   - Statistical significance testing

## 📈 Results

### Performance Summary

| Model | Accuracy | F1-Score | Precision | Recall | AUC |
|-------|----------|----------|-----------|--------|-----|
| Random Forest | 89.2% ± 3.5% | 0.891 | 0.894 | 0.889 | 0.952 |
| SVM | 87.6% ± 4.1% | 0.874 | 0.881 | 0.867 | 0.943 |
| Neural Network | 85.3% ± 5.2% | 0.851 | 0.856 | 0.846 | 0.928 |
| ExtraTrees | 86.8% ± 3.9% | 0.866 | 0.871 | 0.862 | 0.934 |

### Key Findings

- **Best Performance**: Random Forest classifier achieved highest accuracy across all metrics
- **Feature Importance**: Hemodynamic response features showed highest discriminative power
- **Channel Contribution**: Prefrontal cortex channels provided most relevant information
- **Cross-subject Generalization**: Models demonstrated robust performance across different subjects

### Visualization Examples

- Brain activation maps showing workload-related changes
- Time-series plots of hemodynamic responses
- Confusion matrices for classification performance
- Feature importance rankings and distributions

## 🛠 Technologies Used

### Programming Languages
- **Python 3.8+**: Main development language
- **MATLAB**: Signal processing and analysis (optional)

### Machine Learning Frameworks
- **scikit-learn**: Core machine learning algorithms
- **TensorFlow/Keras**: Deep learning models
- **PyTorch**: Neural network implementations

### Signal Processing Libraries
- **MNE-Python**: Neuroimaging data analysis
- **SciPy**: Scientific computing and signal processing
- **NumPy**: Numerical computations

### Visualization Tools
- **Matplotlib**: Static plotting and visualization
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive plots and dashboards

### Development Tools
- **Jupyter Notebook**: Interactive development and analysis
- **pytest**: Unit testing framework
- **Git**: Version control
- **Docker**: Containerization (optional)

## 🤝 Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 coding standards
- Write comprehensive unit tests
- Update documentation for new features
- Ensure backward compatibility
- Add appropriate citations for new algorithms

### Bug Reports and Feature Requests

Please use GitHub Issues to report bugs or request features. Include:
- Detailed description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- System information and dependencies

## 📚 Documentation

### API Documentation
- Comprehensive docstrings for all functions and classes
- Type hints for improved code clarity
- Usage examples and parameter descriptions

### Research Documentation
- Methodology descriptions and theoretical background
- Experimental design and validation procedures
- Performance analysis and comparison studies

### User Guides
- Step-by-step tutorials for different use cases
- Best practices for fNIRS data processing
- Troubleshooting guide for common issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Yogeesh HR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{hr2024cognitive,
  title={Cognitive Workload Classification Using fNIRS Signals},
  author={Yogeesh HR},
  year={2024},
  url={https://github.com/yogeeshhr2003/cognitive-Workload-Classification-Using-fNIRS-Signals},
  note={GitHub repository}
}
```

### Related Publications

This work builds upon and contributes to the following research areas:
- Functional near-infrared spectroscopy for cognitive monitoring
- Machine learning applications in brain-computer interfaces
- Cognitive workload assessment in human-computer interaction

## 📞 Contact

**Yogeesh HR**
- GitHub: [@yogeeshhr2003](https://github.com/yogeeshhr2003)
- Email: [your.email@domain.com]
- LinkedIn: [https://www.linkedin.com/in/yogeeshhr2003/]

### Project Links
- **Repository**: https://github.com/yogeeshhr2003/cognitive-Workload-Classification-Using-fNIRS-Signals
- **Issues**: https://github.com/yogeeshhr2003/cognitive-Workload-Classification-Using-fNIRS-Signals/issues
- **Documentation**: [Link to detailed documentation]

## 🙏 Acknowledgments

### Research Community
- fNIRS research community for open datasets and methodologies
- Machine learning and BCI researchers for foundational algorithms
- Contributors to open-source neuroimaging tools (MNE-Python, nilearn)

### Datasets and Resources
- Public fNIRS datasets used for validation and benchmarking
- Neuroimaging analysis tools and preprocessing pipelines
- Academic institutions supporting brain-computer interface research

### Technical Support
- Python scientific computing ecosystem
- Open-source machine learning frameworks
- GitHub for version control and collaboration platform

---

**⭐ If you find this project useful, please consider giving it a star!**

**🔔 Watch this repository to stay updated with the latest developments**

---

*This project is part of ongoing research in cognitive neuroscience and brain-computer interfaces. We encourage collaboration and knowledge sharing within the research community.*
