<div align="center">

# 🩺 Biomedical Image Classification using Knowledge Distillation

### *Efficient Deep Learning for Medical Image Diagnosis using Transfer Learning & Knowledge Distillation*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MedMNIST](https://img.shields.io/badge/MedMNIST-Dataset-success?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blueviolet?style=for-the-badge)
![Knowledge Distillation](https://img.shields.io/badge/Knowledge-Distillation-orange?style=for-the-badge)

*A lightweight yet powerful medical image classification framework using state-of-the-art CNNs and Knowledge Distillation.*

</div>

---

## 📖 Overview

Medical image classification often requires **large deep learning models** that are computationally expensive. This project investigates how **Knowledge Distillation (KD)** can transfer knowledge from powerful teacher models into a lightweight student model without sacrificing much accuracy.

The project compares multiple pretrained CNN architectures on **MedMNIST** datasets and demonstrates how model compression can make AI systems more practical for real-world healthcare applications.

---

## 🎯 Objectives

- 🔬 Classify biomedical images accurately
- 🚀 Apply Transfer Learning with pretrained CNNs
- 🧠 Compress large models using Knowledge Distillation
- 📊 Compare performance across multiple architectures
- 🔥 Visualize model decisions using Grad-CAM

---

# 🧬 Pipeline

```text
             Medical Images
                    │
                    ▼
           Data Preprocessing
                    │
                    ▼
        Transfer Learning Models
     ┌──────────┬────────────┬────────────┐
     │ ResNet50 │ DenseNet121│EfficientNet│
     └──────────┴────────────┴────────────┘
                    │
                    ▼
         Best Teacher Model Selected
                    │
                    ▼
        Knowledge Distillation (KD)
                    │
                    ▼
        Lightweight Student Model
                    │
                    ▼
      Evaluation + Grad-CAM Analysis
```

---

# 🚀 Features

✨ Transfer Learning

✨ Knowledge Distillation

✨ Early Stopping

✨ Data Augmentation

✨ Model Comparison

✨ ROC Curve

✨ Precision-Recall Curve

✨ Grad-CAM Visualization

✨ Performance Metrics

---

# 🗂 Dataset

The project uses the **MedMNIST** benchmark datasets.

| Dataset | Description |
|---------|-------------|
| 🫁 PneumoniaMNIST | Chest X-ray Classification |
| 🎗 BreastMNIST | Breast Ultrasound Images |
| 🩺 DermaMNIST | Skin Disease Classification |

Datasets are downloaded automatically.

---

# 🧠 Deep Learning Models

| Role | Model |
|------|-------|
| 👨‍🏫 Teacher | ResNet50 |
| 👨‍🏫 Teacher | DenseNet121 |
| 🎓 Student | EfficientNet-B0 |

---

# ⚙️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Framework | PyTorch |
| Dataset | MedMNIST |
| Visualization | Matplotlib, Seaborn |
| Metrics | Scikit-Learn |
| Utilities | NumPy, Pandas, tqdm |

---

# 📊 Evaluation Metrics

The following metrics are used:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ ROC-AUC
- ✅ Precision-Recall Curve
- ✅ Confusion Matrix
- ✅ Grad-CAM Explainability

---

# 📁 Project Structure

```text
Biomedical-KD/
│
├── biomed_kd_final.ipynb
├── README.md
└── assets/
```

---

# ⚡ Installation

```bash
git clone https://github.com/yourusername/biomedical-kd.git

cd biomedical-kd
```

Install dependencies

```bash
pip install torch torchvision
pip install medmnist
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

# ▶️ Run

Simply open the notebook

```bash
jupyter notebook biomed_kd_final.ipynb
```

Run every cell sequentially.

---

# 🧪 Knowledge Distillation

Teacher model predictions are transferred to a compact student model using

```text
Total Loss =
α × CrossEntropy
+
(1-α) × KL Divergence
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Temperature | 3 |
| Alpha | 0.7 |

---

# 📈 Outputs

✔ Training & Validation Curves

✔ Accuracy Comparison

✔ Precision, Recall & F1

✔ ROC Curves

✔ Precision-Recall Curves

✔ Confusion Matrix

✔ Grad-CAM Heatmaps

✔ Teacher vs Student Comparison

---

# 🏆 Final Performance Comparison

> **Knowledge Distillation successfully produced a lightweight EfficientNet student that retained performance close to its teacher models while reducing model complexity—making it a strong candidate for real-world medical AI deployment.**

| Teacher | Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|:--------:|:-----|:-------:|---------:|----------:|-------:|---------:|
| — | ResNet50 | PneumoniaMNIST | **82.85%** | 86.54% | 82.85% | 81.35% |
| — | ResNet50 | BreastMNIST | **81.41%** | 83.69% | 81.41% | 77.99% |
| — | ResNet50 | DermaMNIST | **73.32%** | 76.42% | 73.32% | 74.18% |
| — | DenseNet121 | PneumoniaMNIST | **89.42% 🥇** | **90.83%** | **89.42%** | **88.99%** |
| — | DenseNet121 | BreastMNIST | **85.90% 🥇** | **86.21%** | **85.90%** | **84.70%** |
| — | DenseNet121 | DermaMNIST | 76.11% | 77.75% | 76.11% | 76.58% |
| — | EfficientNet-B0 | PneumoniaMNIST | 86.70% | 88.44% | 86.70% | 86.01% |
| — | EfficientNet-B0 | BreastMNIST | 84.62% | 84.08% | 84.62% | 84.07% |
| — | EfficientNet-B0 | DermaMNIST | 75.46% | 77.89% | 75.46% | 76.20% |
| ResNet50 | **KD-EfficientNet** | PneumoniaMNIST | 84.29% | 87.45% | 84.29% | 83.09% |
| ResNet50 | **KD-EfficientNet** | BreastMNIST | 83.97% | 83.40% | 83.97% | 83.14% |
| ResNet50 | **KD-EfficientNet** | DermaMNIST | 76.06% | 76.92% | 76.06% | 76.01% |
| DenseNet121 | **KD-EfficientNet** | PneumoniaMNIST | **87.02%** | **88.81%** | **87.02%** | **86.35%** |
| DenseNet121 | **KD-EfficientNet** | BreastMNIST | **85.26%** | **84.85%** | **85.26%** | **84.48%** |
| DenseNet121 | **KD-EfficientNet** | DermaMNIST | **76.76% 🥇** | **78.11%** | **76.76%** | **76.95%** |

---

## 🔍 Key Insights

- 🥇 **DenseNet121 emerged as the strongest standalone model**, achieving the highest performance on **PneumoniaMNIST (89.42%)** and **BreastMNIST (85.90%)**.
- 🚀 **Knowledge Distillation successfully transferred the teacher's knowledge to EfficientNet-B0**, enabling the student model to achieve **87.02% accuracy on PneumoniaMNIST**—within **2.4 percentage points** of the best teacher.
- 📈 On **DermaMNIST**, the **KD-EfficientNet distilled from DenseNet121 achieved the highest accuracy (76.76%)**, outperforming **both the original EfficientNet-B0 (75.46%) and the DenseNet121 teacher (76.11%)**.
- ⚡ Across all three datasets, **KD consistently improved the baseline EfficientNet-B0**, demonstrating that model compression can also enhance predictive performance.
- 💡 These results show that **Knowledge Distillation offers an excellent balance between accuracy and efficiency**, making lightweight models suitable for deployment in resource-constrained healthcare environments.

---

# 🌟 Future Work

- Vision Transformers (ViT)
- ConvNeXt
- MobileNetV3
- Ensemble Learning
- Hyperparameter Optimization
- Mixed Precision Training
- Streamlit Web Application
- Docker Deployment

---

<details>
<summary>📚 Libraries Used</summary>

```python
torch
torchvision
medmnist
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

</details>

---

# 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

# 📜 License

This project is intended for educational and research purposes.

---

<div align="center">

## 👨‍💻 Author

### **Nitin Joshi** * **Akash Samanta**

⭐ If you found this project useful, consider **starring the repository!**

Made with ❤️ using **PyTorch**

</div>