# Insurance Fraud Detection
https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data

## 📝 Overview
This project implements a machine learning model to detect insurance fraud using a dataset. It explores various data preprocessing techniques, model training, and evaluation. The model employs the Explainable Boosting Machine (EBM) for classification.

이 프로젝트는 데이터셋을 사용하여 보험 사기를 탐지하는 머신러닝 모델을 구현합니다. EDA, 모델 훈련 및 평가를 탐색합니다. 이 모델은 분류를 위해 Explainable Boosting Machine (EBM)을 사용합니다.

### 🚀 Getting Started

## 📦 Requirements
- numpy
- pandas
- seaborn
- matplotlib
- plotly
- scikit-learn
- imbalanced-learn
- interpret` (for explainable AI)

## 📂 Dataset
The dataset used in this project is fraud_oracle.csv, which contains records of insurance claims with some labeled as fraudulent.

이 프로젝트에서 사용된 데이터셋은 fraud_oracle.csv로, 일부가 사기로 레이블이 지정된 보험 청구 기록을 포함하고 있습니다.

## 🔍 Key Features
- Data importation and preprocessing
데이터 가져오기 및 전처리
- T-Test, Chi-square
- Handling imbalanced classes using SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
- SMOTENC(명목형 및 연속형 특성을 위한 합성 소수 샘플링 기법)를 사용한 불균형 클래스 처리
- Model training using Explainable Boosting Classifier
- Explainable Boosting Classifier를 사용한 모델 훈련
- Evaluation metrics: accuracy, precision, recall, and F1-score
평가 지표: 정확도, 정밀도, 재현율 및 F1 점수

## 📊 모델 평가 결과

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.81      | 0.99   | 0.89     | 1113    |
| 1     | 0.71      | 0.11   | 0.19     | 293     |
| **Accuracy**      |           |        | **0.81**     | **1406**    |
| **Macro Avg**     | 0.76      | 0.55   | 0.54     | 1406    |
| **Weighted Avg**  | 0.79      | 0.81   | 0.74     | 1406    |


## ⚙️ Setup Instructions
Clone the repository:

리포지토리를 클론합니다:
bash

git clone https://github.com/yourusername/Insurance-Fraud-Detection.git

cd Insurance-Fraud-Detection
