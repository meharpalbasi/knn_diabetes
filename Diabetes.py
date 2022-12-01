#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


df = pd.read_csv('diabetes.csv')
df


# In[11]:


X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].values
y = df["Outcome"].values
print(X.shape, y.shape)


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5, stratify=y)
knn= KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))


# In[14]:


len(X_train)


# In[16]:


len(X_test)


# In[5]:


train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1,30)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)


# In[6]:


plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
#Testing accuracy highest at k=6


# In[23]:


knn = KNeighborsClassifier(n_neighbors = 6)


# In[24]:


knn.fit(X_train,y_train)


# In[25]:


y_pred = knn.predict(X_test)
y_pred


# In[26]:


#cHECK THE ACCURACY SCORE AGAIN:
from sklearn.metrics import accuracy_score


# In[27]:


accuracy_score(y_test,y_pred)


# In[36]:


#WE RECEIVE DATA FOR 3 NEW INDIVIDUALS
new_patients = pd.read_csv('Diabetes_new_patients.csv')


# In[39]:


X_new = new_patients.to_numpy()


# In[42]:


predictions=knn.predict(X_new)


# In[43]:


print('Predictions: {}'.format(predictions))


# In[ ]:


#Only the first subject would have diabetes

