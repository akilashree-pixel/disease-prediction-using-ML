

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier



def load_heart_disease():
    """UCI Heart Disease Dataset (Cleveland)"""
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(url, names=cols, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df

def load_diabetes():
    """UCI Pima Indians Diabetes Dataset"""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    cols = ["pregnancies","glucose","bloodPressure","skinThickness",
            "insulin","BMI","diabetesPedigree","age","target"]
    df = pd.read_csv(url, names=cols)
    return df

def load_breast_cancer_data():
    """Sklearn Breast Cancer Dataset (UCI)"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

datasets = {
    "Heart Disease" : load_heart_disease(),
    "Diabetes"      : load_diabetes(),
    "Breast Cancer" : load_breast_cancer_data(),
}



models = {
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"                : SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost"            : XGBClassifier(n_estimators=100, use_label_encoder=False,
                                         eval_metric="logloss", random_state=42),
}



all_results = {}   # {dataset_name: {model_name: metrics_dict}}

for ds_name, df in datasets.items():
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    all_results[ds_name] = {}

    
    print(f"  Dataset : {ds_name}  |  Samples: {len(df)}  |  Features: {X.shape[1]}")


    for mdl_name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        y_pred      = mdl.predict(X_test_s)
        y_prob      = mdl.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        metrics = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test, y_pred, zero_division=0),
            "f1"       : f1_score(y_test, y_pred, zero_division=0),
            "auc"      : auc(fpr, tpr),
            "cm"       : confusion_matrix(y_test, y_pred),
            "fpr"      : fpr,
            "tpr"      : tpr,
            "y_test"   : y_test,
            "y_pred"   : y_pred,
        }
        all_results[ds_name][mdl_name] = metrics

        print(f"\n  ── {mdl_name}")
        print(f"     Accuracy : {metrics['accuracy']:.4f}")
        print(f"     Precision: {metrics['precision']:.4f}")
        print(f"     Recall   : {metrics['recall']:.4f}")
        print(f"     F1 Score : {metrics['f1']:.4f}")
        print(f"     AUC      : {metrics['auc']:.4f}")



COLORS  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
DS_COLS = {"Heart Disease": "#e74c3c", "Diabetes": "#8e44ad", "Breast Cancer": "#e91e8c"}


for ds_name, res in all_results.items():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(f"{ds_name} — Model Performance", fontsize=15, fontweight="bold")
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for ax, metric, label in zip(axes, metric_names, labels):
        vals  = [res[m][metric] for m in models]
        bars  = ax.bar(list(models.keys()), vals, color=COLORS, edgecolor="white", width=0.6)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(list(models.keys()), rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{ds_name.replace(' ','_')}_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

for ds_name, res in all_results.items():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"{ds_name} — Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, (mdl_name, metrics) in zip(axes, res.items()):
        sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=1, linecolor="white",
                    xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
        ax.set_title(mdl_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{ds_name.replace(' ','_')}_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()


for ds_name, res in all_results.items():
    plt.figure(figsize=(7, 6))
    for (mdl_name, metrics), color in zip(res.items(), COLORS):
        plt.plot(metrics["fpr"], metrics["tpr"], color=color, lw=2,
                 label=f"{mdl_name}  (AUC = {metrics['auc']:.3f})")
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"{ds_name} — ROC Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{ds_name.replace(' ','_')}_roc.png", dpi=150, bbox_inches="tight")
    plt.show()


acc_matrix = pd.DataFrame(
    {ds: {m: all_results[ds][m]["accuracy"] for m in models} for ds in datasets}
)
plt.figure(figsize=(8, 4))
sns.heatmap(acc_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=0.5, vmin=0.7, vmax=1.0,
            annot_kws={"size": 11, "weight": "bold"})
plt.title("Accuracy Heatmap — All Models × All Datasets",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("accuracy_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


for ds_name, df in datasets.items():
    X = df.drop("target", axis=1)
    y = df["target"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(StandardScaler().fit_transform(X), y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    top = importances[-10:]    # top 10

    plt.figure(figsize=(8, 4))
    bars = plt.barh(top.index, top.values, color=DS_COLS[ds_name], edgecolor="white")
    plt.xlabel("Importance", fontsize=11)
    plt.title(f"{ds_name} — Top Feature Importances (Random Forest)",
              fontsize=12, fontweight="bold")
    for bar, val in zip(bars, top.values):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{ds_name.replace(' ','_')}_feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.show()

print("\n✅ All done! Charts saved as PNG files.")