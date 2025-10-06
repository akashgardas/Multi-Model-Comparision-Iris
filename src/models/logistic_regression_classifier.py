from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def run_logreg(X_train, X_test, y_train, y_test):
    """
    Logistic Regression classifier for Iris dataset.
    Suitable for multi-class problems using multinomial strategy.
    """
    model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Purples', fmt='d', ax=ax)
    ax.set_title("Logistic Regression - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return acc, report, fig
