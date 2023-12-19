import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=23)

print(iris.data.shape)
print('Training Set',len(x_train))
print('Test Set',len(x_test))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)


y_pred = knn.predict(x_test)
print(y_pred)


from sklearn import metrics
print("Accuracy = ",metrics.accuracy_score(y_pred,y_test))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
