# Understanding Dropout Regularisation in Neural Networks  
### A Tutorial Using the Fashion-MNIST Dataset

This repository accompanies the tutorial and report that explore the **effectiveness of dropout regularisation** in neural networks. By training two multilayer perceptron (MLP) models — one with dropout and one without — this project demonstrates how dropout influences **overfitting**, **generalisation**, and **training stability**.

The work is based on the analysis and results presented in the attached report *“Understanding Dropout Regularisation in Neural Networks”* by Srijan.  
The full report is included as:  

---

## 📌 Project Overview

This project includes:

- A clear conceptual explanation of **dropout**  
- Mathematical formulation of the dropout mechanism  
- Implementation of two neural network models (with and without dropout)  
- Training and validation curves for both models  
- Comparison of final test accuracy and loss  
- Discussion of overfitting, stability, and generalisation  
- Ethical AI considerations related to model robustness  
- A complete Jupyter notebook to reproduce all results  

The repository is structured to serve as an **educational tutorial**, helping learners understand why dropout works and when it is most effective.

---

## 📊 Dataset

**Dataset:** Fashion-MNIST  
Source: TensorFlow Keras Datasets  
Link: https://github.com/zalandoresearch/fashion-mnist  

The dataset contains:

- 60,000 training images  
- 10,000 test images  
- 28×28 grayscale fashion-category images  
- 10 balanced classes  

In the notebook, the dataset is loaded automatically using:

```python
from tensorflow.keras.datasets import fashion_mnist
```

#### ⚠️ No manual dataset download is required, so the repository does not contain a /data folder.

## Methods Used
#### 1. Neural Network Models

Two MLP models were implemented:

- Model A: No dropout

- Model B: Uses dropout (rate = 0.5)

Each model uses:

- Flatten layer

- Dense(256, ReLU)

- Dense(128, ReLU)

- Optional Dropout(0.5)

- Dense(10, Softmax)

#### 2. Optimisation

- Optimizer: Adam

- Loss: Sparse categorical crossentropy

- Epochs: 20

- Batch size: 128

#### 3. Metrics

The following were evaluated for both models:

- Training loss curves

- Validation loss curves

- Training accuracy

- Validation accuracy

- Final test accuracy

- Final test loss

Figures generated include:

- Training & validation loss curves (with vs. without dropout)

- Training & validation accuracy curves

- Bar chart comparing final metrics

All figures appear in the /figures directory.

### 🛠️ How to Run the Notebook

1. Install dependencies
```
pip install tensorflow numpy matplotlib seaborn
```

2. Open the notebook
```
ML_Assignment.ipynb
```
Running the notebook will automatically:

1. Load the Fashion-MNIST dataset

2. Train both models

3. Generate all plots

4. Print final test metrics

### 📁 Folder Structure
```
project/
│
├── figures/
│   ├── training_loss_no_dropout.png
│   ├── training_loss_dropout.png
│   ├── accuracy_curves.png
│   ├── metric_comparison.png
│   └── ...
│
├── dropout_tutorial.ipynb
├── Srijan_ML_Report.pdf / .docx
├── README.md
└── LICENSE
```
### 📈 Key Results

Results reproduced from the report:

Test Accuracy

- No Dropout: 0.896

- Dropout 0.5: 0.889

### Interpretation

- Dropout slightly reduced test accuracy in this experiment.

- However, dropout made the model more stable and less prone to overfitting.

- The dataset subset and model architecture were not highly overfitting, so dropout’s benefit was limited.

- This behaviour is consistent with deep learning theory, where dropout is most effective in deeper or highly expressive networks.

### 🧠 Key Insights from the Report

- Dropout works by randomly deactivating neurons, forcing the network to learn redundant representations.

- It acts as a form of implicit model averaging.

- Small datasets or shallow networks may not benefit significantly.

- Dropout helps avoid co-adaptation of neurons.

- Ethically, dropout contributes to model robustness, which reduces unpredictable failures during deployment.

### 📚 References

Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.

Goodfellow, Bengio, Courville (2016). Deep Learning.

Fashion-MNIST Dataset (Zalando Research)

TensorFlow Keras Documentation

(All references also appear in the report.)

### 📄 License

This project is released under the MIT License.
See the LICENSE file for details.

### 🙌 Acknowledgements

Zalando Research for the Fashion-MNIST dataset

TensorFlow / Keras for deep learning tools

University instructors for module guidance
