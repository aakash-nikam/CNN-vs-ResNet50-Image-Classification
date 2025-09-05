CNN vs ResNet50: A Comparative Study on Product Image Classification

Author: Akash Nikam (DBS – MSc Data Analytics)
Tools: Python (TensorFlow/Keras, scikit-learn)

Project Overview

This project compares two deep learning approaches for a five-class retail product image classification task:

A custom Convolutional Neural Network (CNN) built from scratch.

A fine-tuned ResNet50 model using transfer learning from ImageNet.

The dataset includes 1,738 labeled images across five categories: Background, Product 1, Product 2, Product 3, and Product 4. The objective is to evaluate how a model trained from scratch performs versus a transfer learning approach, particularly under class imbalance and limited data conditions.

Tech and Libraries

Python 3.x

TensorFlow/Keras

scikit-learn

matplotlib, seaborn

Install requirements with:

pip install tensorflow scikit-learn matplotlib seaborn

Dataset

Images are organized into five folders (one per class).

Data split: 70% training / 30% validation.

Preprocessing included rescaling (0–1), resizing to 224×224 pixels, and augmentation (rotation, flip, zoom, shifts, shear).

Models
Custom CNN

Three Conv2D layers (32, 64, 128 filters)

MaxPooling after each conv layer

Dense(128, ReLU) + Dropout(0.5)

Output softmax layer (5 classes)

Optimizer: Adam, Loss: categorical crossentropy, Epochs: 10

ResNet50 Transfer Learning

Pretrained on ImageNet

Added head: GAP → Dense(128, ReLU) → Dropout(0.5) → Softmax

Frozen base layers initially

Fine-tuned last 40 layers with reduced learning rate

Results
Metric	CNN	ResNet50 (fine-tuned)
Training Accuracy	96.27%	94.88%
Validation Accuracy	70.75%	63.50%
Validation Loss	2.37	0.89
Recall (Product 1)	27%	37%
Recall (Product 2)	0%	12.5%
Weighted F1-score	0.66	0.60
Overfitting	High	Minimal

CNN reached higher accuracy but overfit strongly.

ResNet50 generalized better, achieving lower validation loss.

Product 2 remained the hardest class due to imbalance.

Visuals

Stored in results/:

Confusion Matrices (CNN, ResNet50)

Training vs Validation Accuracy/Loss plots

How to Run

Clone the repo and place dataset under data/ with one subfolder per class.

Run Final code.ipynb (Jupyter Notebook).

Outputs include confusion matrices, accuracy/loss plots, and classification reports.

Future Work

Apply class reweighting or SMOTE to handle imbalance.

Use early stopping and LR scheduling to stabilize training.

Explore newer architectures (EfficientNet, Vision Transformers).

Try ensembles of CNN + ResNet50.

Academic Artefacts

Full report: reports/Akash_Nikam_20054691__Himanshu Dubey_20027763.pdf

Contact

Akash Nikam – MSc Data Analytics, Dublin Business School
Email: aakashn3118@gmail.com
