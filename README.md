# Description

This repository is part of my coursework for the MMDT (Myannar Data Tech) program under Dr. Myo Thida, focusing on project-based learning in Deep Learning.
It includes all assignments, experiments, and applied projects developed throughout the course; covering topics from fundamental neural network architectures to advanced model optimization and analysis.
The goal of this repository is to demonstrate practical mastery of deep learning principles, model experimentation, and performance evaluation using real-world datasets and problem statements.

# Introduction To Deep Learning

This repo includes all the example codes and datasets used in my book 'Introduction to Deep Learning'. The purpose of this book is to document my teachings in a physical form and make it accessible for students with limited resources to learn from. The book aims to provide a comprehensive and easy-to-follow introduction to the fundamental concepts of deep learning methods.

I believe that hands-on learning is crucial for understanding, and thus, the explanations in the book are accompanied by detailed Python code snippets throughout the text. Readers can follow the instructions and run the code in this repository on their own computer or on an online platform such as Google Colab.

---
## Deep Learning Project – Housing Price Prediction (Singapore HDB Resale)
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

**Evaluation Metrics**
- R² Score
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

<img width="940" height="467" alt="image" src="https://github.com/user-attachments/assets/2355c1ff-1e67-4914-960e-f5e44cccf082" />

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

---

## Deep Learning Project – Myanmar Village Name Region Classification
**Chapter 2 — Project 2: Multi-Class Text Classification with Deep Learning**

**Project Overview**

This project tackles a challenging multi-class text classification problem — predicting a Myanmar administrative region/state based solely on Romanized village names.
The task is inspired by real-world geocoding challenges in low-resource languages, where name variations and inconsistent spellings make classification highly complex.

The project demonstrates skills in:
- Natural Language Processing (NLP) with deep learning architectures
- Feature engineering and text normalization for noisy, low-resource data
- Model experimentation with multiple encoding and embedding strategies
- Performance evaluation using statistical tests and visualization

**Objective**

- Develop and evaluate multiple deep learning models to classify Romanized Burmese village names into 18 regional classes by:
- Comparing various text encoding and tokenization techniques
- Testing different deep learning architectures (Dense, LSTM, Conv1D, hybrid)
- Handling imbalanced and ambiguous data through class weighting and stratified sampling
- Evaluating the statistical significance of name–region relationships

**Dataset**

Source: Custom dataset combining MIMU, local news, and administrative data
Input: Romanized Burmese village names
Target: Administrative state/region (18 classes)
Size: Several thousand labeled entries

**Data Challenges**

- Duplicate names: Some villages appear in multiple regions → label ambiguity
- Short text inputs: Many names < 5 characters → weak contextual signal
- Imbalanced data: Major regions overrepresented → model bias risk

# Methodology

**1. Data Cleaning & Preprocessing**

- Removed noisy characters (parentheses, slashes) using regex
- Unified spellings with phonetic normalization to reduce dialect-based variation
- Merged aspirated/compound consonants (e.g., ph, hp, hs → p, s)
- Normalized vowels/diphthongs (au, aw, ay → o, e)
- Simplified endings (aung → ong, ein → en)
- Lowercased and removed redundant letters (pp → p, ll → l)
- Verified meaningful correlation between names and regions via Chi-square test (p < 0.05)

**2. Encoding and Tokenization Techniques**

| Method                                      | Description               | Pros                                                 | Limitations                             |
| ------------------------------------------- | ------------------------- | ---------------------------------------------------- | --------------------------------------- |
| **One-Hot Encoding**                        | Baseline representation   | Simple, interpretable                                | Sparse, ignores character relationships |
| **TF-IDF (considered)**                     | Frequency-based weighting | Works for long texts                                 | Ineffective for short names             |
| **Character-Level Tokenization**            | Each char → integer       | Captures subword variation                           | Limited context                         |
| **N-Gram Tokenization (TextVectorization)** | Bigrams/trigrams          | Models local dependencies                            | Larger vocab, higher sparsity           |
| **FastText Embeddings**                     | Trained on dataset        | Learns subword-level semantics, handles unseen names | Requires more computation               |

**3. Training Strategy**

To ensure robust learning:
  - Applied class weighting to handle imbalanced classes
  - Used stratified sampling for fair data splits
  - Prevented data leakage by separating test data before preprocessing
  - Added dropout, batch normalization, and early stopping for regularization

**Model Architectures Tested**
| Model ID    | Architecture                       | Encoding   | Train Acc (%) | Test Acc (%) | Remarks                          |
| ----------- | ---------------------------------- | ---------- | ------------- | ------------ | -------------------------------- |
| Baseline    | 2 Hidden Layers                    | One-hot    | 84            | 28           | Overfits severely                |
| Model 1     | 3 Hidden Layers + Dropout + BN     | One-hot    | 87            | 25           | Regularization not effective     |
| Model 2     | Embedding → BiLSTM → GlobalMaxPool | Char-level | 60            | 25           | Struggles with short names       |
| Model 3     | Deep BiLSTM (stacked)              | Char-level | 45            | 22           | Vanishing gradients              |
| Model 4     | Conv1D + BiLSTM                    | Char-level | 40            | 27           | Moderate improvement             |
| Model 5     | Conv1D + LSTM                      | Bigram     | 77            | 26           | Better training, still overfits  |
| Model 6     | Conv1D + LSTM                      | Trigram    | 69            | 24           | Overcomplex, less generalization |
| Model 7     | Conv1D + Dense Layers              | FastText   | 95            | 38           | Major performance jump           |
| Model 8     | Conv1D + Dense Layers (100 epochs) | FastText   | 92            | 40           | Best generalization              |
| Model 9     | Conv1D + LSTM + Dense Layers       | FastText   | 95            | 40           | Best-performing model            |
| Model 10–11 | Phonetic Norm + FastText           | FastText   | 92–95         | 35–37        | Slight performance drop          |

**Key Results**
Best Model:
  - Conv1D + LSTM + FastText Embeddings (Epochs = 130)
  - Test Accuracy: 40%
  - Training Accuracy: 95%
  - Balanced prediction coverage across regions (see confusion matrix)
  - Reduced bias toward dominant regions
  - Shows improved generalization over baseline and char-level tokenization

**Confusion Matrix Insights**

Baseline Model:
  - High bias toward frequent classes
  - Widespread misclassification on rare regions
Best Model (FastText + ConvLSTM):
  - Improved diagonal dominance in confusion matrix
  - Better minority class recognition
  - Stronger prediction spread across multiple regions

**Statistical Validation**

A Chi-square test of independence (p < 0.05) confirmed a statistically significant relationship between village names and regions, validating that the classification problem is non-random and learnable despite complexity.

**💡 Findings & Insights**

- Encoding drives performance: Character-level or one-hot representations fail to capture subword semantics; FastText embeddings outperform all.
- Model depth ≠ accuracy: Deep BiLSTMs struggled with vanishing gradients due to very short inputs.
- CNN + LSTM hybrids capture both local n-gram features and sequential patterns, improving generalization.
- Overfitting persists: Limited, imbalanced data and lack of contextual features hinder generalization.
- Statistical confirmation: The p-value < 0.05 validates feasibility of using text-based inference for geographic prediction.

**Reflection & Learnings**

“This project reshaped how I view NLP challenges in low-resource languages.”
- Learned to design encoding strategies for short, noisy Romanized text.
- Understood how phonetic normalization helps reduce spelling inconsistencies.
- Recognized that CNNs can effectively extract local character-level patterns in text (not just for images).
- Gained hands-on experience in feature engineering, class imbalance handling, and evaluation metrics (e.g., Chi-square test).
- Realized the importance of statistical validation before deep learning modeling.

This project strengthened my ability to combine linguistic insight, statistical reasoning, and deep learning architecture design to solve challenging real-world classification problems.

**🏁 Conclusion**

This project demonstrates advanced deep learning applications for language-based geolocation prediction in low-resource linguistic environments.
Despite data limitations, the best model achieved significant improvements using FastText embeddings and hybrid CNN–LSTM architectures, setting a strong foundation for future research on Myanmar language NLP and geographical text classification.

---
