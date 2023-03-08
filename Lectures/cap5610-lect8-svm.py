#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines
# * In this demo, we will train and test SVMs on the Iris dataset. 
# * For illustration purpose, we will only use the sepal length and sepal width as features.
# * The full version of this demo is available here (try this as an exercise): https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

# ### 1. Load dataset
# * We can use the function load_iris to load the dataset. 
# * The API for this function is at: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

# In[ ]:


from sklearn.datasets import load_iris

iris = load_iris()
print(iris.DESCR) # Print the description of the dataset

# Get the inputs and labels from the dataset
X = iris.data[:, :2] # We restrict to the sepal features
Y = iris.target


# ### 2. Visualize dataset
# 
# Now we plot a scatter plot with one color for each class. We have seen this in the previous demo.

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


# ### 3. Train/test split
# 
# Now we split the data into a train set (75%) and a test set (25%).

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# ### 4. Train an SVM classifier on the train set
# * We train an SVM classifier on the train set. The API for SVM is: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# * Here there are 3 options with 3 different kernels. 
# * Uncomment the model that you want to train (and comment out the others).

# In[ ]:


from sklearn import svm

C = 1.0
model = svm.SVC(kernel='linear', C=C)
# model = svm.SVC(kernel='rbf', gamma=0.7, C=C)
# model = svm.SVC(kernel='poly', degree=4, gamma='auto', C=C)

model.fit(X_train, Y_train)


# ### 5. Evaluate the trained SVM on the test set
# We use the trained SVM to make predictions on the test set and compute the accuracy.

# In[ ]:


from sklearn.metrics import accuracy_score

Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print('Accuracy on test set:', acc)


# ### 6. Visualize the trained SVM
# 
# To visualize the decision boundary of a classifier with 2d inputs, we first need to create a [mesh grid](https://www.geeksforgeeks.org/numpy-meshgrid-function/) on the 2d space covering all inputs (from min to max values of each dimension).

# In[ ]:


import numpy as np

X0 = X[:, 0]    # Sepal length feature
X1 = X[:, 1]    # Sepal width feature
h = 0.02        # Stepsize for meshgrid

# Find the range of the 2 dimensions that we will plot
X0_min, X0_max = X0.min() - 1, X0.max() + 1
X1_min, X1_max = X1.min() - 1, X1.max() + 1

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(X0_min, X0_max, h),
                     np.arange(X1_min, X1_max, h))

print(xx)
print(yy)


# Next, we need to make a prediction with our SVM model on every point on the grid.

# In[ ]:


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print(Z)


# Finally, we plot the contour of the predictions together with all the data points.

# In[ ]:


import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=100, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

