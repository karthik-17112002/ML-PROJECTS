import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("D://Mall_Customers.csv")
print(df)
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
x=df.iloc[:,[3,4]].values
print(x)
from sklearn.cluster import KMeans
wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
plt.plot(range(1,11),wcss_list)
plt.title("the elbow method graph")
plt.xlabel("number of clusters")
plt.ylabel("wcss_list")
plt.show()
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_predict=kmeans.fit_predict(x)


plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="blue",label='cluster=1')
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c="red",label='cluster=2')
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c='yellow',label="cluster=3")
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],s=100,c='orange',label="cluster=4")
plt.scatter(x[y_predict==4,0],x[y_predict==4,1],s=100,c="magenta",label="cluster=5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label="centroid")
plt.title("cluster of customers")
plt.legend()

