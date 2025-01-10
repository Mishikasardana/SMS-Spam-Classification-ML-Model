# SMS Spam Classification

## Overview
This project involves building an AI model that classifies SMS messages as either **spam** or **legitimate** (non-spam). The classification is achieved by converting text messages into numerical features using techniques such as **TF-IDF** or **word embeddings**, and applying machine learning algorithms such as **Naive Bayes**, **Logistic Regression**, or **Support Vector Machines (SVMs)**.

The primary goal is to accurately distinguish spam messages from legitimate ones, ensuring minimal false positives and false negatives. This project uses Python and leverages libraries like **scikit-learn**, **NLTK**, and **pandas** for data preprocessing, model training, and evaluation.

---

## Table of Contents
1. [Features](#features)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Requirements](#requirements)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Contributing](#contributing)

---

## Features
- Text preprocessing using **TF-IDF** and **word embeddings**.
- Application of machine learning algorithms:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machines (SVMs)
- Evaluation metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## Dataset
The dataset contains labeled SMS messages indicating whether each message is spam or legitimate. The typical structure includes:
- **Text:** The content of the SMS.
- **Label:** A binary variable where:
  - `1` = Spam
  - `0` = Legitimate (Non-Spam)

You can use publicly available datasets like the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## Methodology
1. **Data Preprocessing:**
   - Cleaning and tokenizing text messages using **NLTK**.
   - Converting text into numerical features using **TF-IDF** or word embeddings.
   - Splitting data into training and testing sets.

2. **Model Training:**
   - Train models using algorithms such as:
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machines (SVMs)

3. **Evaluation:**
   - Assess model performance using metrics such as accuracy, precision, recall, and F1-Score.

---

## Requirements
Install the required libraries:
```bash
pip install numpy pandas scikit-learn nltk
```


## Evaluation
The model will be evaluated using the following metrics:
- **Accuracy:** Percentage of correctly classified messages.
- **Precision:** Proportion of correctly predicted spam messages out of all predicted spam messages.
- **Recall:** Proportion of actual spam messages correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.


---

## Results
- **[nltk_data] Downloading package stopwords to /root/nltk_data...**
- **[nltk_data]   Package stopwords is already up-to-date!**
- **Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='object')**
1. **Naive Bayes Classifier:**
- **Accuracy: 0.9721973094170404**
- **Precision: 1.0**
- **Recall: 0.7933333333333333**
- **F1 Score: 0.8847583643122676**

2. **Logistic Regression Classifier:**
- **Accuracy: 0.9506726457399103**
- **Precision: 0.9702970297029703**
- **Recall: 0.6533333333333333**
- **F1 Score: 0.7808764940239044**

3. **Support Vector Machine Classifier:**
- **Accuracy: 0.9757847533632287**
- **Precision: 0.984251968503937**
- **Recall: 0.8333333333333334**
- **F1 Score: 0.9025270758122743**

---

## Contributing
Contributions are welcome! If you have ideas for improvements or new features, follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork.
4. Open a pull request for review.

---
