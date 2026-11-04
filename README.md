# Online Shopper Purchase Intention Prediction Using ANN

## Project Overview

This project develops an **Artificial Neural Network (ANN)** to predict whether an online shopping session will result in a **Purchase** or **No Purchase**.

The model uses visitor browsing behavior, page interaction, traffic information, and visitor characteristics to estimate purchase intention.

### Target Variable

- **0 / False → No Purchase**
- **1 / True → Purchase**

The project covers:

- Data Cleaning
- Exploratory Data Analysis
- Data Preprocessing
- ANN Model Building
- Early Stopping
- Class Weight Experiment
- Classification Threshold Tuning
- Model Evaluation
- Model Saving

---

## Dataset

The **Online Shoppers Purchasing Intention Dataset** originally contained:

- **12,330 rows**
- **18 columns**

During data cleaning:

- No missing values were found.
- **125 duplicate rows** were removed.

Final cleaned dataset:

- **12,205 rows**
- **18 columns**


## Dataset Features

| Feature | Description |
|---|---|
| Administrative | Number of administrative pages visited |
| Administrative_Duration | Time spent on administrative pages |
| Informational | Number of informational pages visited |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Number of product-related pages visited |
| ProductRelated_Duration | Time spent on product-related pages |
| BounceRates | Bounce rate associated with visited pages |
| ExitRates | Exit rate associated with visited pages |
| PageValues | Value of pages in relation to completed transactions |
| SpecialDay | Closeness to a special shopping day |
| Month | Month of the session |
| OperatingSystems | Operating system category |
| Browser | Browser category |
| Region | Visitor region category |
| TrafficType | Source/type of website traffic |
| VisitorType | Returning Visitor, New Visitor, or Other |
| Weekend | Whether the session occurred on a weekend |
| Revenue | Target indicating Purchase or No Purchase |

---

## Target Distribution

After cleaning:

| Class | Count | Percentage |
|---|---:|---:|
| No Purchase | 10,297 | 84.37% |
| Purchase | 1,908 | 15.63% |

The dataset is **imbalanced**, with Purchase sessions representing only about **15.6%** of the observations.

Therefore, model performance was evaluated using not only accuracy but also:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

---

## Exploratory Data Analysis

EDA was performed to understand feature distributions, target distribution, correlations, and differences between Purchase and No Purchase sessions.

### Key Findings

Purchase sessions generally showed:

- Higher visitor engagement
- Higher `PageValues`
- Lower `BounceRates`
- Lower `ExitRates`

Important correlations with Revenue included:

| Feature | Correlation with Revenue |
|---|---:|
| PageValues | 0.492 |
| ExitRates | -0.204 |
| BounceRates | -0.145 |

Some input features were also strongly correlated:

- `BounceRates` and `ExitRates` → approximately **0.902**
- `ProductRelated` and `ProductRelated_Duration` → approximately **0.860**

Categorical features such as `Month`, `VisitorType`, and `TrafficType` also showed differences in purchase behavior.

---

## Data Preparation

The target variable `Revenue` was separated from the input features.

Boolean variables were converted to numerical form:

- `False → 0`
- `True → 1`

The `Weekend` feature was also converted to 0 and 1.

The dataset was then divided approximately into:

- **Training Set:** 64%
- **Validation Set:** 16%
- **Test Set:** 20%

Stratified splitting was used to maintain the Purchase/No Purchase distribution across the datasets.

The test set was kept separate from model training and model selection.

---

## Data Preprocessing

Numerical features were standardized using **StandardScaler**.

The following categorical features were processed using **OneHotEncoder**:

- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType

Although `OperatingSystems`, `Browser`, `Region`, and `TrafficType` contain numbers, these numbers represent categories rather than numerical quantities.

A **ColumnTransformer** was used to combine numerical scaling and categorical encoding.

The preprocessor was fitted only on the training data to prevent data leakage.

After preprocessing, the ANN received:

**73 input features**

---

# ANN Model

## Architecture

The ANN architecture was:

**73 Inputs → 16 Neurons → 8 Neurons → 1 Output**

The model used:

- **First hidden layer:** 16 neurons with ReLU
- **Second hidden layer:** 8 neurons with ReLU
- **Output layer:** 1 neuron with Sigmoid

Total trainable parameters:

**1,329**

### Dense Layers

Dense layers are fully connected layers in which each neuron receives information from the previous layer.

They were used to allow the ANN to learn relationships and interactions between the shopper features.

### ReLU

**ReLU** was used in the hidden layers to introduce non-linearity, allowing the network to learn more complex relationships in shopper behavior.

### Sigmoid

The output layer used **Sigmoid** because this is a binary classification problem.

It produces a value between approximately 0 and 1 representing the predicted probability of the **Purchase** class.

---

## Model Compilation

The ANN was compiled using:

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Metric:** Accuracy

### Binary Cross-Entropy

Binary Cross-Entropy was used because the target contains two classes:

- 0 → No Purchase
- 1 → Purchase

It measures the difference between the actual classes and the probabilities predicted by the ANN.

### Adam Optimizer

Adam controls how the ANN updates its weights while minimizing the loss.

It was selected because it provides efficient and adaptive weight updates and is widely used for neural-network training.

---

## Model Training

The initial model was trained with:

- **Maximum Epochs:** 50
- **Batch Size:** 32

An **epoch** represents one complete pass of the training data through the ANN.

A **batch size of 32** means that the ANN processes 32 training observations before performing a weight update.

---

# Baseline ANN

The baseline ANN achieved approximately:

| Metric | Result |
|---|---:|
| Validation Loss | 0.2892 |
| Validation Accuracy | 88.12% |

The training curves indicated that the training performance continued improving while validation performance eventually stopped improving.

This indicated **overfitting**, so Early Stopping was introduced.

---

# Early-Stopping ANN

Early Stopping was configured using:

- `monitor='val_loss'`
- `patience=5`
- `restore_best_weights=True`

### Why Early Stopping?

Early Stopping monitors validation loss while the ANN is training.

If validation loss stops improving for five epochs, training is stopped and the weights from the best-performing epoch are restored.

This helps prevent the ANN from continuing to train after it begins overfitting.

## Performance

The Early-Stopping ANN achieved:

| Metric | Result |
|---|---:|
| Validation Loss | **0.2604** |
| Validation Accuracy | **88.58%** |
| ROC-AUC | **0.9085** |
| Purchase Precision | **0.65** |
| Purchase Recall | **0.58** |
| Purchase F1-score | **0.61** |

### Confusion Matrix

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| No Purchase | 1552 | 96 |
| Purchase | 127 | 178 |

The model correctly identified **178 of 305 actual purchasers**, but missed **127**.

Although the model achieved strong ROC-AUC, the Purchase recall of approximately **58%** indicated that many actual purchasers were still being missed.

---

# Class Weight Experiment

Because the dataset is imbalanced, a second ANN was trained using **class weights**.

Calculated weights:

- **No Purchase (0): 0.5926**
- **Purchase (1): 3.1986**

The minority Purchase class received a larger weight so that errors involving purchasers had more influence during ANN training.

The same **16 → 8 → 1 architecture** and Early Stopping configuration were retained.

## Class-Weighted ANN Performance

| Metric | Result |
|---|---:|
| Validation Loss | **0.3870** |
| Validation Accuracy | **84.84%** |
| ROC-AUC | **0.8991** |
| Purchase Precision | **0.51** |
| Purchase Recall | **0.76** |
| Purchase F1-score | **0.61** |

### Confusion Matrix

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| No Purchase | 1425 | 223 |
| Purchase | 73 | 232 |

Class weighting increased Purchase recall from **58% to 76%** and reduced missed purchasers from **127 to 73**.

However, it also reduced precision and increased false-positive predictions.

---

# Model Comparison

| Metric | Early-Stopping ANN | Class-Weighted ANN |
|---|---:|---:|
| Validation Accuracy | **88.58%** | 84.84% |
| Purchase Precision | **0.65** | 0.51 |
| Purchase Recall | 0.58 | **0.76** |
| Purchase F1-score | 0.61 | 0.61 |
| ROC-AUC | **0.9085** | 0.8991 |
| True Positives | 178 | **232** |
| False Negatives | 127 | **73** |
| False Positives | **96** | 223 |

Class weighting improved Purchase recall considerably, but the Purchase F1-score remained approximately unchanged.

The **Early-Stopping ANN without class weights** retained better precision, accuracy, and ROC-AUC.

Therefore, the Early-Stopping ANN was retained for **classification threshold tuning**.

---

# Classification Threshold Tuning

The ANN produces a probability for the Purchase class.

The default classification threshold is:

**0.50**

Therefore:

`Probability >= 0.50 → Purchase`

`Probability < 0.50 → No Purchase`

At the default threshold, the Early-Stopping ANN achieved only about **58% Purchase recall**.

Different thresholds are therefore evaluated using the **validation set**.

Lowering the threshold generally:

- Increases Purchase recall
- Decreases Purchase precision

The threshold is selected based on the **Purchase-class F1-score** to find a better balance between precision and recall.

> **Note:** Threshold tuning must be rerun using the latest Early-Stopping ANN before the final threshold is selected.

---

# Final Model Evaluation

The final **Early-Stopping ANN without class weights**, using a classification threshold of **0.23**, was evaluated on the previously untouched test set.

The model achieved the following results:

| Metric | Result |
|---|---:|
| Test Accuracy | **87%** |
| Purchase Precision | **0.57** |
| Purchase Recall | **0.77** |
| Purchase F1-Score | **0.65** |
| ROC-AUC | **0.9222** |

### Confusion Matrix

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| **No Purchase** | 1832 | 227 |
| **Purchase** | 87 | 295 |

The model correctly identified **295 out of 382 actual purchasers**, achieving a Purchase recall of approximately **77%**. It missed **87 actual purchasers**.

The ROC-AUC score of **0.9222** indicates that the model has a strong ability to distinguish between Purchase and No Purchase sessions.

Overall, the final ANN provides a good balance between identifying actual purchasers and maintaining reasonable overall classification performance.


# Conclusion

An **Artificial Neural Network (ANN)** was developed to predict online shopper purchase intention.

The baseline ANN showed signs of overfitting, so **Early Stopping** was introduced. The Early-Stopping ANN achieved a validation ROC-AUC of approximately **0.9085**, indicating strong ability to distinguish between Purchase and No Purchase sessions.

Because the dataset is imbalanced, **class weighting** was also tested. It increased Purchase recall from **58% to 76%**, but reduced precision and increased false-positive predictions. The Purchase F1-score remained approximately unchanged.

Therefore, the **Early-Stopping ANN without class weights** was retained, and classification threshold tuning was performed using the validation set. A threshold of **0.23** was selected to improve the balance between Purchase precision and recall.

Using the selected threshold, the final model achieved:

- **Test Accuracy:** 87%
- **Purchase Precision:** 0.57
- **Purchase Recall:** 0.77
- **Purchase F1-Score:** 0.65
- **ROC-AUC:** 0.9222

The model correctly identified **295 out of 382 actual purchasers**, achieving a Purchase recall of approximately **77%**.

Overall, the final **ANN with Early Stopping, no class weights, and a classification threshold of 0.23** demonstrated good performance and strong generalization to unseen data.


