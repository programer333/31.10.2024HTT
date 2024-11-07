import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

cart_clf_iris = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_clf_iris.fit(X_train_iris, y_train_iris)
y_pred_cart_iris = cart_clf_iris.predict(X_test_iris)

id3_clf_iris = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_clf_iris.fit(X_train_iris, y_train_iris)
y_pred_id3_iris = id3_clf_iris.predict(X_test_iris)

print("CART (Gini) - IRIS Accuracy:", accuracy_score(y_test_iris, y_pred_cart_iris))
print("CART (Gini) - IRIS Report:\n", classification_report(y_test_iris, y_pred_cart_iris))
print("ID3 (Entropy) - IRIS Accuracy:", accuracy_score(y_test_iris, y_pred_id3_iris))
print("ID3 (Entropy) - IRIS Report:\n", classification_report(y_test_iris, y_pred_id3_iris))


X_train_dental, X_test_dental, y_train_dental, y_test_dental = train_test_split(X_dental, y_dental, test_size=0.2, random_state=42)

cart_clf_dental = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_clf_dental.fit(X_train_dental, y_train_dental)
y_pred_cart_dental = cart_clf_dental.predict(X_test_dental)

id3_clf_dental = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_clf_dental.fit(X_train_dental, y_train_dental)
y_pred_id3_dental = id3_clf_dental.predict(X_test_dental)

print("CART (Gini) - Dental Accuracy:", accuracy_score(y_test_dental, y_pred_cart_dental))
print("CART (Gini) - Dental Report:\n", classification_report(y_test_dental, y_pred_cart_dental))
print("ID3 (Entropy) - Dental Accuracy:", accuracy_score(y_test_dental, y_pred_id3_dental))
print("ID3 (Entropy) - Dental Report:\n", classification_report(y_test_dental, y_pred_id3_dental))