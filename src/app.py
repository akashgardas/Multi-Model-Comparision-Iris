import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import classifier modules
import models.knn_classifier as knn_classifier
import models.svm_classifier as svm_classifier
import models.bayes_classifier as bayes_classifier
import models.logistic_regression_classifier as logistic_regression_classifier

# Load data once
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

st.set_page_config(page_title="Iris Classifier Comparison", layout="wide")
st.title("🌸 Multi-Model Comparison on Iris Dataset")
st.markdown("Explore and compare machine learning classifiers on the classic Iris dataset.")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["📊 Dataset Visualization", "🤖 Run Classifiers"])

# Split data once for all models
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
#  Visualization Section
# =========================
if menu == "📊 Dataset Visualization":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Pairplot of Iris Dataset")
    fig1 = sns.pairplot(df, hue="species", diag_kind="kde")
    st.pyplot(fig1)

    st.subheader("Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Boxplots by Feature")
    fig3, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, feature in enumerate(iris.feature_names):
        sns.boxplot(x='species', y=feature, data=df, ax=axes[i])
        axes[i].set_title(feature)
    st.pyplot(fig3)

# =========================
#  Classifier Section
# =========================
elif menu == "🤖 Run Classifiers":
    st.subheader("Select a Classifier to Evaluate")
    model_choice = st.selectbox(
        "Choose a model:",
        ["KNN", "SVM", "Naive Bayes", "Logistic Regression"]
    )

    if st.button("Run Model"):
        if model_choice == "KNN":
            acc, report, fig = knn_classifier.run_knn(X_train_scaled, X_test_scaled, y_train, y_test)
        elif model_choice == "SVM":
            acc, report, fig = svm_classifier.run_svm(X_train_scaled, X_test_scaled, y_train, y_test)
        elif model_choice == "Naive Bayes":
            acc, report, fig = bayes_classifier.run_bayes(X_train_scaled, X_test_scaled, y_train, y_test)
        elif model_choice == "Logistic Regression":
            acc, report, fig = logistic_regression_classifier.run_logreg(X_train_scaled, X_test_scaled, y_train, y_test)

        st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")
        st.text("Classification Report:")
        st.text(report)
        st.pyplot(fig)
