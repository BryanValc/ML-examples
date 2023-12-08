

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')

def plot_roc_curve(fpr, tpr, title, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadero Positivo')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresión Logística
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print("Precisión Regresión Logística:", acc_logreg)

# SVM (Máquinas de Soporte Vectorial)
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Precisión SVM:", acc_svm)

# Árbol de Decisión
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print("Precisión Árbol de Decisión:", acc_tree)

# Regresión Logística - Visualización
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plot_confusion_matrix(cm_logreg, 'Matriz de Confusión - Regresión Logística')

fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
plot_roc_curve(fpr_logreg, tpr_logreg, 'Curva ROC - Regresión Logística', roc_auc_logreg)

# SVM - Visualización
cm_svm = confusion_matrix(y_test, y_pred_svm)
plot_confusion_matrix(cm_svm, 'Matriz de Confusión - SVM')

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
plot_roc_curve(fpr_svm, tpr_svm, 'Curva ROC - SVM', roc_auc_svm)

# Árbol de Decisión - Visualización
cm_tree = confusion_matrix(y_test, y_pred_tree)
plot_confusion_matrix(cm_tree, 'Matriz de Confusión - Árbol de Decisión')

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_tree)
roc_auc_tree = auc(fpr_tree, tpr_tree)
plot_roc_curve(fpr_tree, tpr_tree, 'Curva ROC - Árbol de Decisión', roc_auc_tree)

