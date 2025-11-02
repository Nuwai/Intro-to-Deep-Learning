<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/2355c1ff-1e67-4914-960e-f5e44cccf082" /><img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/0e7f42e0-275e-472c-9be3-3440de6d3ad7" /># Description

This repository is part of my coursework for the MMDT (Myannar Data Tech) program under Dr. Myo Thida, focusing on project-based learning in Deep Learning.
It includes all assignments, experiments, and applied projects developed throughout the course; covering topics from fundamental neural network architectures to advanced model optimization and analysis.
The goal of this repository is to demonstrate practical mastery of deep learning principles, model experimentation, and performance evaluation using real-world datasets and problem statements.

# Introduction To Deep Learning

This repo includes all the example codes and datasets used in my book 'Introduction to Deep Learning'. The purpose of this book is to document my teachings in a physical form and make it accessible for students with limited resources to learn from. The book aims to provide a comprehensive and easy-to-follow introduction to the fundamental concepts of deep learning methods.

I believe that hands-on learning is crucial for understanding, and thus, the explanations in the book are accompanied by detailed Python code snippets throughout the text. Readers can follow the instructions and run the code in this repository on their own computer or on an online platform such as Google Colab.

---
## Chapter 2 — Project 1: Deep Neural Network Regression
**Project Overview**
This project develops and optimizes Deep Neural Network (DNN) regression models to predict HDB resale housing prices in Singapore using TensorFlow.
It demonstrates practical deep learning workflows — from data preprocessing and model architecture design to hyperparameter tuning, regularization, and performance evaluation.
The project is part of my Deep Learning course portfolio, showcasing applied skills in:
- Neural network architecture optimization
- Model generalization and overfitting prevention
- Evaluation metric interpretation
- Trade-off analysis between performance and computational cost

**Dataset**
- Source: HDB Resale Prices Dataset (Data.gov.sg)
- Period Covered: Jan 2017 – Mar 2024
- Size: 180,154 records, 11 columns
- Target Variable: adjusted_price (normalized housing resale price)
- Features: Structural and demographic attributes such as floor area, flat type, lease remaining, and floor number.
- 
# Methodology
**1. Data Preparation**

- Cleaned and standardized dataset.
- No missing data detected (government-maintained dataset).
- Split into train/test (70/30).
- StandardScaler applied to numerical features.

**2. Model Development**
Built and compared multiple architectures using TensorFlow/Keras:
| Model               | Hidden Layers | Regularization                        | Activation | Epochs | Batch Size | Early Stopping |
| ------------------- | ------------- | ------------------------------------- | ---------- | ------ | ---------- | -------------- |
| Baseline            | 2             | None                                  | ReLU       | 10     | 32         | No             |
| Three HL            | 3             | None                                  | ReLU       | 10     | 32         | No             |
| Four HL             | 4             | None                                  | ReLU       | 50     | 32         | No             |
| Regularized Model 1 | 4             | BatchNorm + Dropout (all layers) + L2 | ReLU       | 100    | 32         | Yes              |
| Regularized Model 2 | 4             | BatchNorm + Dropout (2nd layer) + L2  | ReLU       | 100    | 32         | Yes              |
| Leaky ReLU          | 4             | BatchNorm + Dropout + L2              | Leaky ReLU | 100    | 32         | Yes              |
Evaluation Metrics

**R² Score**

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Training Time (seconds)

**📈 Experimental Results**

| Model               | R² (Train) | R² (Test) | RMSE   | MAE    | MAPE (%) | Params | Time (s) |
| ------------------- | ---------- | --------- | ------ | ------ | -------- | ------ | -------- |
| Baseline            | 0.82       | 0.82      | 86,256 | 64,777 | 10.81    | 2,209  | 114.5    |
| Three HL            | 0.90       | 0.90      | 64,218 | 46,230 | 7.44     | 5,953  | 115.7    |
| Four HL             | 0.93       | 0.93      | 55,285 | 39,879 | 6.52     | 17,537 | 405.8    |
| Regularized Model 1 | 0.93       | 0.93      | 55,259 | 39,332 | 6.36     | 18,433 | 323.9    |
| Regularized Model 2 | 0.92       | 0.91      | 59,125 | 42,559 | 6.90     | 6,529  | 282.3    |
| Leaky ReLU          | 0.93       | 0.93      | 54,665 | 39,187 | 6.36     | 18,433 | 416.9    |

**Findings & Insights**

- Model depth improves performance — up to a point. Beyond 4 layers, gains plateau while computation time rises sharply.
- Regularization techniques (BatchNorm + Dropout + L2) successfully reduced overfitting and improved generalization.
- Regularized Model 2 offered the best trade-off:
- Strong test performance (R² = 0.91, MAPE = 6.9)
- ~30% less training time vs. deeper models
- Compact architecture (6,529 parameters)
- Leaky ReLU provided comparable accuracy but at higher computational cost.
- Interpretation: Optimal architectures balance depth and efficiency, not just raw accuracy.

**Key Technical Learnings**

- Implemented feedforward neural networks for regression with TensorFlow/Keras.
- Tuned architectural hyperparameters and regularization strategies.
- Performed comparative model analysis with visualizations (R² vs Training Time).
- Applied early stopping for optimal convergence and generalization.
- Used custom evaluation functions integrating sklearn.metrics and tabulated reporting.

  Model Comparison — R² vs Training Time
  

**🏁 Conclusion**
Regularized Model 2 achieved the best generalization with an R² of 0.91, RMSE ≈ 59K, and MAPE ≈ 6.9%, while maintaining reasonable computational efficiency.
The experiment illustrates key machine learning trade-offs and practical deep learning optimization for real-world tabular data.
