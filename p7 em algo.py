import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
print(x)
x.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']


plt.figure(figsize=(14,7))

colormap = np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.xlabel("Sepal Length(cm)")
plt.ylabel("Sepal Width(cm)")
plt.scatter(x.Sepal_Length,x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')



model = KMeans(n_clusters=3, random_state = 100)

model.fit(x)

y_pred = model.labels_
print(y_pred)


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state = 100)

gmm.fit(x)
y_cluster_gmm = gmm.predict(x)
y_cluster_gmm

plt.figure(figsize=(16,7))

plt.subplot(1,3,1)
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y.Targets],s=40)
plt.title('Actual Classification')

plt.subplot(1,3,2)
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y_pred],s=40)
plt.title('K Mean Classification')

plt.subplot(1,3,3)
plt.xlabel("Petal Length(cm)")
plt.ylabel("Petal Width(cm)")
plt.scatter(x.Petal_Length,x.Petal_Width, c=colormap[y_cluster_gmm],s=40)
plt.title('GMM Classification')


print('K Means Accuracy')
print('Accuracy', sm.accuracy_score(y,y_pred))
print('Confusion Matrix \n', sm.confusion_matrix(y,y_pred))

print('GMM Accuracy')
print('Accuracy', sm.accuracy_score(y,y_cluster_gmm))
print('Confusion Matrix \n', sm.confusion_matrix(y,y_cluster_gmm))
