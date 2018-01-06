print(__doc__)

import numpy as np
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: clf
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

accuracy = float(np.sum(preds == y_test))/y_test.shape[0]
print("accuracy:", accuracy)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))

print("\n-----------------------------\n")

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

accuracy = float(np.sum(preds == y_test))/y_test.shape[0]
print("accuracy:", accuracy)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, preds))
