# SkinDerm U-Net: Skin Lesion Segmentation using U-Net Architecture

A deep learning project for automated skin lesion segmentation using the U-Net convolutional neural network architecture. This project aims to assist in the analysis and diagnosis of dermatological conditions through precise segmentation of dermoscopy images.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project implements a U-Net deep learning model for segmenting skin lesions from dermoscopy images. Accurate segmentation is a crucial first step in computer-aided diagnosis (CAD) systems for skin cancer detection and classification.

**Key Objectives:**
- Automated segmentation of skin lesions from dermoscopic images
- High accuracy boundary detection for clinical analysis
- Efficient preprocessing and augmentation pipeline
- Model evaluation with medical imaging metrics

## ✨ Features

- **U-Net Architecture**: Implementation of the classic U-Net model optimized for medical image segmentation
- **Data Preprocessing**: Comprehensive image preprocessing pipeline including normalization, resizing, and augmentation
- **Training Pipeline**: Complete training workflow with validation and checkpointing
- **Evaluation Metrics**: Implementation of medical imaging metrics (Dice coefficient, IoU, Sensitivity, Specificity)
- **Visualization Tools**: Utilities for visualizing predictions and comparing with ground truth masks
- **Model Inference**: Easy-to-use inference scripts for new dermoscopy images

## 📊 Dataset

The model is trained on dermoscopy image datasets containing:
- High-resolution skin lesion images
- Corresponding binary segmentation masks
- Various skin types and lesion categories

**Supported Datasets:**
- ISIC (International Skin Imaging Collaboration)
- PH2 Dataset
- Custom dermoscopy datasets

**Data Format:**
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1_mask.png
    ├── image2_mask.png
    └── ...
```

## 🏗️ Model Architecture

The U-Net architecture consists of:
- **Encoder (Contracting Path)**: Captures context through convolutional and max-pooling layers
- **Bottleneck**: Deepest layer with maximum channel depth
- **Decoder (Expanding Path)**: Enables precise localization through upsampling and skip connections
- **Skip Connections**: Concatenates features from encoder to decoder for better gradient flow

**Model Specifications:**
- Input Size: 256x256x3 (configurable)
- Output: Binary segmentation mask
- Loss Function: Binary Cross-Entropy + Dice Loss
- Optimizer: Adam
- Total Parameters: ~31M (standard U-Net)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shakti019/SkinDerm-Unet-.git
cd SkinDerm-Unet-
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training

1. **Prepare your dataset:**
   - Organize images and masks in the required structure
   - Update data paths in the configuration file

2. **Start training:**
```python
python train.py --config config.yaml --epochs 100 --batch-size 16
```

### Inference

**Single Image Prediction:**
```python
python predict.py --image path/to/image.jpg --model checkpoints/best_model.pth --output output/
```

**Batch Prediction:**
```python
python predict_batch.py --input-dir images/ --model checkpoints/best_model.pth --output-dir results/
```

### Evaluation

```python
python evaluate.py --test-data test/ --model checkpoints/best_model.pth
```

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| Dice Coefficient | 0.XX |
| IoU (Jaccard Index) | 0.XX |
| Sensitivity (Recall) | 0.XX |
| Specificity | 0.XX |
| Accuracy | 0.XX |

### Sample Predictions

| Original Image | Ground Truth | Prediction |
|---------------|--------------|------------|
| ![Input](docs/sample1_input.jpg) | ![GT](docs/sample1_gt.png) | ![Pred](docs/sample1_pred.png) |

## 📁 Project Structure

```
SkinDerm-Unet-/
├── Derm project/           # Main project directory
│   ├── data/              # Dataset directory
│   ├── models/            # Model architecture definitions
│   ├── utils/             # Utility functions
│   ├── notebooks/         # Jupyter notebooks for experimentation
│   ├── checkpoints/       # Saved model weights
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   ├── evaluate.py        # Evaluation script
│   └── config.yaml        # Configuration file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 📦 Requirements

```
tensorflow>=2.6.0
keras>=2.6.0
numpy>=1.19.5
opencv-python>=4.5.3
matplotlib>=3.4.3
scikit-learn>=0.24.2
scikit-image>=0.18.3
Pillow>=8.3.2
tqdm>=4.62.3
pyyaml>=5.4.1
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Training parameters
- Augmentation settings

```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  image_size: 256

model:
  input_channels: 3
  num_classes: 1
  base_filters: 64

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  early_stopping_patience: 15
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- **ISIC Archive**: For providing dermoscopy image datasets
- **Medical Imaging Community**: For research and best practices in medical image segmentation

## 📞 Contact

**Shakti019** - [@Shakti019](https://github.com/Shakti019)

Project Link: [https://github.com/Shakti019/SkinDerm-Unet-](https://github.com/Shakti019/SkinDerm-Unet-)

---

## 📚 References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. ISIC Archive: https://www.isic-archive.com/
3. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection", IEEE 2017

## 🔮 Future Work

- [ ] Implementation of U-Net++ and Attention U-Net variants
- [ ] Multi-class segmentation for different lesion types
- [ ] Integration with classification models for end-to-end diagnosis
- [ ] Mobile deployment using TensorFlow Lite
- [ ] Web application for easy model access
- [ ] Ensemble methods for improved robustness

---

⭐ If you find this project helpful, please consider giving it a star!
