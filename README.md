# disease-prediction-using-ML
ML-based disease prediction project using Heart Disease, Diabetes, and Breast Cancer datasets. Compares multiple classifiers and visualizes model performance with ROC curves, confusion matrices, and feature importance

> A machine learning project that predicts the likelihood of diseases using structured patient medical data. 

---

## 📌 Objective

To predict the possibility of diseases based on patient data by applying multiple classification algorithms on real-world medical datasets from the UCI Machine Learning Repository.

---

## 📂 Datasets Used

| Dataset | Source | Samples | Features |
| ❤️ Heart Disease | UCI ML Repository | 303 | 13 |
| 🩸 Diabetes | Pima Indians (UCI) | 768 | 8 |
| 🔬 Breast Cancer | UCI / Sklearn | 569 | 30 |

---

## 🤖 Algorithms Implemented

- ✅ Random Forest Classifier
- ✅ Support Vector Machine (SVM)
- ✅ Logistic Regression
- ✅ XGBoost Classifier

---

## 📊 Visualizations

- Bar charts for Accuracy, Precision, Recall, and F1 Score
- Confusion Matrices for all models
- ROC Curves with AUC scores
- Cross-dataset Accuracy Heatmap
- Feature Importance plots (Random Forest)

---

## 🛠️ Tech Stack

`Python` `Scikit-learn` `XGBoost` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

**2. Install dependencies**
```bash
pip install scikit-learn xgboost seaborn pandas matplotlib
```

**3. Run the script**
```bash
python disease_prediction.py
```
Or open `disease_prediction.ipynb` in Jupyter Notebook / Google Colab.

---

## 📈 Results Summary

| Model | Heart Disease | Diabetes | Breast Cancer |
| Random Forest | ~87% | ~79% | ~96% |
| SVM | ~83% | ~77% | ~95% |
| Logistic Regression | ~80% | ~75% | ~93% |
| XGBoost | ~89% | ~82% | ~97% |

---

## 📁 Project Structure

```
disease-prediction/
│
├── disease_prediction.py       
├── disease_prediction.ipynb    
├── README.md                   
└── outputs/
    ├── Heart_Disease_metrics.png
    ├── Heart_Disease_confusion.png
    ├── Heart_Disease_roc.png
    ├── Diabetes_metrics.png
    ├── Diabetes_confusion.png
    ├── Diabetes_roc.png
    ├── Breast_Cancer_metrics.png
    ├── Breast_Cancer_confusion.png
    ├── Breast_Cancer_roc.png
    ├── accuracy_heatmap.png
    └── *_feature_importance.png
```

---

## 🙋‍♀️ Author

** Akila shree M ** — Machine Learning Intern @ CodeAlpha  
📎 [LinkedIn](https://linkedin.com) · 💻 [GitHub](https://github.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
