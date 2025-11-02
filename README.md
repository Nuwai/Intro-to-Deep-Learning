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
- Noisy text variation: Inconsistent Romanization styles due to dialect differences.

**Phonetic Normalization**

Romanized Burmese text often varies due to regional pronunciation.
Normalization rules included:
  - Reducing aspirated/compound consonants (ph, hp → p)
  - Standardizing vowels (au, aw, ay → o or e)
  - Unifying tone endings (aung → ong, ein → en)
  - Removing duplicate letters (pp → p)
  - Lowercasing and cleaning text
  - This ensured the model could generalize better across spelling and dialect variations.
    
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
Observation:
- Deep LSTMs underperformed due to sparse input.
- CNN + FastText captured most subword-level variations effectively.
- Despite improvements, models still overfitted, suggesting strong dataset bias.

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
Best Model (Conv1D + LSTM + FastText):
  - Improved diagonal dominance in confusion matrix
  - Better minority class recognition
  - Less overfitting, though accuracy still modest (~40%).

**Statistical Validation**

A Chi-square test of independence (p < 0.05) confirmed a statistically significant relationship between village names and regions, validating that the classification problem is non-random and learnable despite complexity.

**💡 Findings & Insights**

- Encoding drives performance: Character-level or one-hot representations fail to capture subword semantics; FastText embeddings outperform all.
- Model depth ≠ accuracy: Deep BiLSTMs struggled with vanishing gradients due to very short inputs.
- CNN + LSTM hybrids capture both local n-gram features and sequential patterns, improving generalization.
- Overfitting persists: Limited, imbalanced data and lack of contextual features hinder generalization.
- Statistical confirmation: The p-value < 0.05 validates feasibility of using text-based inference for geographic prediction.

**Learning Points & Analysis**

Bias in Data Collection
  - The dataset was partially collected from local news, which disproportionately features villages in conflict or major events.
  - As a result, the dataset lacks representation from peaceful or remote areas, making it non-inclusive.
  - The model thus overfits to frequent “newsworthy” names, learning patterns that don’t generalize across all regions.

Impact of Short and Sparse Inputs
  - Many village names are only 1–2 words long, giving minimal signal for classification.
  - This caused models, especially LSTMs, to underperform.

Overfitting due to Bias and Limited Variation
  - The small and biased dataset leads to memorization rather than pattern learning.
  - Future data collection must ensure balanced sampling from official geographic datasets (e.g., MIMU, GAD lists).

CNNs for Text
  - Learned that CNNs can effectively capture local character dependencies — not just for images, but also for text classification tasks.

**Reflection & Learnings**

“This project reshaped how I view NLP challenges in low-resource languages.”
- This project deepened understanding of encoding choices, data validation, and bias impact on model behavior.
- Learned that model architecture alone can’t fix biased or incomplete data — data diversity is essential.
- Reinforced the importance of statistical testing before modeling to confirm problem feasibility.
- Realized that deeper networks are not always better when data is short, sparse, or noisy.
This project strengthened my ability to combine linguistic insight, statistical reasoning, and deep learning architecture design to solve challenging real-world classification problems.

**🏁 Conclusion**

This project demonstrates advanced deep learning applications for language-based geolocation prediction in low-resource linguistic environments.
Despite data limitations, the best model achieved significant improvements using FastText embeddings and hybrid CNN–LSTM architectures, setting a strong foundation for future research on Myanmar language NLP and geographical text classification.

**Recommendations for Future Work**

- Expand dataset with more balanced coverage across all states/regions.
- Apply phonetic algorithms (e.g., Soundex, Metaphone) for grouping similar names.
- Explore Transformer-based models fine-tuned on Romanized Burmese.
- Use data augmentation and multi-label classification for duplicated names.
- Combine rule-based and neural methods to leverage both linguistic and statistical features.

---

## CNN Models Comparison and Analysis
### Project Overview

This project evaluates the performance of several Convolutional Neural Network (CNN) architectures — ResNet50, VGGNet16, InceptionV3, ConvNeXt, and EfficientNet — on diverse image types to study their robustness, inference efficiency, and adaptability to domain-specific (Myanmar cultural) data.
The goal is to understand how well pretrained ImageNet models generalize to both standard and culturally unique image inputs under varying conditions.

**Objectives**
- Compare CNN architectures using top-1 and top-3 accuracy metrics.
- Evaluate performance under different image conditions — clear, blurry, noisy, and culturally specific.
- Analyze inference time vs. accuracy trade-offs for real-world deployment.
- Study domain generalization gaps, especially for non-Western (Myanmar) imagery.

**Experimental Setup**
Models Evaluated
- ResNet50
- VGGNet16
- InceptionV3
- ConvNeXtTiny
- EfficientNetB7

All models were loaded with pretrained ImageNet weights for comparative inference testing.

**Dataset**

The test images were manually collected and categorized into three groups:
- Simple and everyday objects
- Noisy, blurry, or visually ambiguous images
- Culturally specific Myanmar images (e.g., pagodas, monks, traditional objects)
  ⚠️ Note: Since the dataset was curated from local news and online sources, it was biased toward more visually striking or conflict-related images. This imbalance led to   overfitting toward familiar object patterns and limited cultural inclusivity in model predictions.

**Code Implementation**

**Preprocessing and Evaluation Pipeline**

Each model requires its own preprocessing pipeline (e.g., normalization and input resizing). A key implementation detail was dynamically assigning the correct preprocessing function for each model architecture (e.g., resnet_preprocess, inception_preprocess, etc.), which significantly improved performance consistency.
```
# Example
if name == 'InceptionV3':
    x = inception_preprocess(x)
elif name == 'ResNet50':
    x = resnet_preprocess(x)
elif name == 'VGGNet16':
    x = vgg_preprocess(x)
elif name == 'EfficientNet':
    x = effnet_preprocess(x)
```
Predictions were made using Top-K decoding, and inference time was recorded for each model.

**Results and Findings**

**🔹 Initial Observations**
- Without correct preprocessing, InceptionV3 underperformed dramatically, producing confident but incorrect predictions.
- After applying the appropriate model-specific preprocessing, accuracy improved to 85–96% across all models.

### 📊 Normalized Evaluation Framework for Top-1 Predictions

| **Model**      | **Avg. Confidence (%)** | **Avg. Time (s)** | **High Confidence (%)** | **Overall Score (Normalized)** |
|----------------|-------------------------|-------------------|--------------------------|--------------------------------|
| **ResNet50**   | 55.0%                   | 0.066             | 19.0%                   | **95.7**                       |
| **InceptionV3**| 50.7%                   | 0.061             | 19.0%                   | **86.9**                       |
| **EfficientNet**| 53.9%                  | 0.399             | 14.3%                   | **46.8**                       |
| **VGGNet16**   | 38.4%                   | 0.080             | 9.5%                    | **36.2**                       |
| **ConvNeXt**   | 46.3%                   | 0.297             | 19.0%                   | **26.8**                       |

Inference speed vs. accuracy trade-off is an essential consideration for real-time systems.

**CNN Robustness on Noisy, Blurry, and Confusing Images**

**Key Observations**
- Top-3 predictions often revealed semantically correct alternatives even when Top-1 was incorrect.
- Example:
  - “Dog wearing sunglasses” → sunglasses, sunglass, dog_breed
  - “Wine glass with flower” → goblet, red_wine, vase
- Misclassifications were semantically reasonable, e.g., camouflage gun → screw, hook, revolver.
  - → Indicates that CNNs generalize visually similar patterns.
- EfficientNet had the highest robustness (Top-1 = 20/33 correct), followed by InceptionV3 and ConvNeXt.

**Domain Gap: Cultural and Out-of-Distribution (OOD) Analysis**

To test generalization beyond ImageNet, models were evaluated on Myanmar cultural imagery, such as pagodas and monks.
- **Key Findings**
  - All models showed low confidence (Top-3 median 0.08–0.14), confirming poor generalization on culturally specific data.
  - The top-1 confidence averaged only 38–55%, reflecting domain mismatch between Myanmar imagery and ImageNet’s Western-centric dataset.

**Qualitative Analysis: Cultural Images**
🛕 Pagoda Images
- Most models predicted “stupa”, “mosque”, or “palace” — semantically similar but not culturally precise.
- Predictions like “punching bag” or “lemon” for golden pagodas reflected shape bias and data scarcity for Southeast Asian contexts.

🙏 Monk Image
- Models mostly focused on the umbrella, predicting “umbrella”, “notebook”, or “kimono”.
- Some outliers (“cellular telephone”, “cobra”) suggest confusion under unusual object combinations.

**Insights and Discussion**
- Top-3 evaluation provides a more realistic measure of model understanding than Top-1 accuracy alone.
- Preprocessing correctness can drastically affect model accuracy — architectural awareness matters.
- Inference time and confidence must be balanced for deployment.
- Out-of-distribution failure reveals cultural bias in ImageNet-trained models, emphasizing the need for localized datasets.
- Data collection bias (focusing on more visible/conflict-related images) led to overfitting and limited inclusivity, reinforcing the importance of diverse sampling in real-world data projects.

**Conclusion**

- CNNs retain semantic awareness under distortion, though Top-1 accuracy can mislead in ambiguous scenarios.
- ResNet50 and InceptionV3 achieved the best balance between accuracy, inference speed, and robustness.
- Evaluating Top-3 predictions offers richer interpretability for ambiguous or culturally diverse datasets.
- The project highlights the importance of fair, inclusive data and the limitations of Western-trained AI models in global contexts.

**Reflection and Learning**

This project deepened our understanding of CNN interpretability, data bias, and preprocessing design.
Key lessons include:
  - Always align preprocessing pipelines with architecture requirements.
  - Evaluation metrics must match real-world ambiguity — top-3 predictions reveal more than top-1 accuracy.
  - Dataset inclusivity matters: collecting from local sources improved cultural relevance but introduced bias toward frequent/conflict imagery, reducing model generalization.
  - Efficiency–accuracy trade-offs are vital in model selection for real-world applications.
Overall, this project strengthened our ability to conduct structured deep learning evaluations and interpret model behavior beyond raw accuracy numbers.
---

