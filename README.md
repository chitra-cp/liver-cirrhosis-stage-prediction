
# Liver Cirrhosis Stage Prediction using Machine Learning

This project focuses on predicting the **stage of liver cirrhosis** using patient health indicators from clinical data. The prediction helps in early diagnosis and better treatment planning.

## 🔍 Problem Statement

Liver cirrhosis is a chronic disease that progresses through multiple stages. Early identification of the stage using biochemical and clinical markers can assist healthcare providers in prioritizing treatment strategies.

## 📁 Dataset

- **File**: `dataset.csv`
- **Size**: 25,000 rows × 19 columns
- **Features include**:
  - Demographics: Age, Sex
  - Symptoms: Ascites, Edema, Hepatomegaly, Spiders
  - Lab Results: Bilirubin, Cholesterol, Albumin, Copper, SGOT, Platelets, etc.
  - Target: `Stage` (categorical, indicating cirrhosis stage)

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 🧠 Models Used & Performance

| Model              | Accuracy | ROC-AUC | Log Loss |
|-------------------|----------|---------|----------|
| XGBoost Classifier| 96.36%   | 0.995   | 0.12     |
| Random Forest      | 95.48%   | 0.993   | 0.20     |
| Logistic Regression| 56.42%   | 0.74    | 0.92     |

**Best Model**: XGBoost with 96.36% accuracy and ROC-AUC score of 0.995.

## 📈 Visualizations

- Correlation heatmap
- Feature importance plots
- Confusion matrices

## 📂 Files

```
📦 liver-cirrhosis-stage-prediction
├── liver_cirrhosis.csv
├── liver_cirrhosis.py
├── README.md
```

## 🚀 How to Use

1. Clone this repository:
```bash
git clone https://github.com/Chitrarajalakshmi/liver-cirrhosis-stage-prediction.git
```
2. Run the script:
```bash
python liver_cirrhosis.py
```
3. Make sure the required libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 👩‍💻 Author

**Chitra S**  
_M.Sc. Data Analytics Graduate_
