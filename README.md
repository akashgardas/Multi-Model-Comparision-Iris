# 🌸 Multi-Model-Comparison-Iris

A comparative analysis of four machine learning classifiers on the classic **Iris dataset** using Python.  
This project aims to visualize, train, and evaluate multiple models — **KNN, SVM, Naive Bayes, and Logistic Regression** — and compare their performance through various metrics and visualizations.

---

## 📂 Project Overview

The **Iris dataset** is a foundational dataset in machine learning, containing 150 samples of iris flowers divided into three species:
- *Setosa*
- *Versicolor*
- *Virginica*

Each sample includes four features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 🎯 Objective

To test and compare the performance of different classification algorithms on the same dataset.  
This helps in understanding model behavior, strengths, and limitations for structured, well-separated data.

---

## ⚙️ Classifiers Used

| Model | Type | Nature | Common Usage |
|--------|------|--------|---------------|
| **K-Nearest Neighbors (KNN)** | Non-parametric | Distance-based | Simple, effective baseline |
| **Support Vector Machine (SVM)** | Margin-based | Kernel-driven | Strong for linear & non-linear data |
| **Naive Bayes** | Probabilistic | Based on Bayes theorem | Fast and interpretable |
| **Logistic Regression** | Linear | Statistical | Interpretable and efficient baseline |

---

## 📊 Evaluation Metrics

Each model is evaluated using:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Log Loss

---

## 📈 Visualizations

- **Pairplot:** Visualize feature separability across species  
- **Heatmap:** Feature correlations  
- **Boxplots:** Distribution per feature  
- **PCA Plot / 3D Scatter:** Multi-dimensional view  
- **Metric Comparison Graphs:** Compare model performances side by side

---

## 🧰 Technologies Used

- Python 🐍  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---
## 📘 Results

| Model               | Accuracy | F1-Score | Remarks                      |
| ------------------- | -------- | -------- | ---------------------------- |
| KNN                 | ~97%     | High     | Sensitive to scaling         |
| SVM                 | ~98%     | High     | Great margin separation      |
| Naive Bayes         | ~94%     | Moderate | Assumes feature independence |
| Logistic Regression | ~96%     | High     | Performs well on linear data |

---

## 📚 Learning Outcomes

Understand classifier comparison using a single dataset

Explore data visualization and preprocessing techniques

Evaluate models using multiple performance metrics

Interpret which models best suit linearly vs. non-linearly separable data

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork this repo, open issues, or submit pull requests for improvements, better visualizations, or additional models.

---

## 📜 License

This project is licensed under the MIT License — feel free to use and modify with credit.

---

## 👤 Author

Akash Gardas
Engineering Student | Machine Learning Enthusiast
🔗 [LinkedIn](https://www.linkedin.com/in/gardas-akash-66102327b/)
📧 Contact: [Email](akash39g@gmail.com)
