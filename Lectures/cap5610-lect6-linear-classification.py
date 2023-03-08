#!/usr/bin/env python
# coding: utf-8

# # CAP5610 - Linear classification
# 
# In this demo, we will train and test a logistic regression model on a toy classification dataset.

# ### 1. Generate the toy dataset
# 
# We can use the [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) function from sklearn to generate the data. The API for this function is at: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html.

# In[ ]:


from sklearn.datasets import make_classification

X, Y = make_classification(n_samples=200,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=2,
                           random_state=10)

print(X.shape, Y.shape)


# ### 2. Visualize the dataset
# 
# We can visualize the dataset by plotting the points with different colors for each class.

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()


# ### 3. Split the data
# 
# Now we split the data into a train set (60%) and a test set (40%).

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

print(X_train.shape)
print(X_test.shape)


# ### 4. Define and train a logistic regression model on the train set
# 
# We define a logistc regression model with l2 regularization and C=1.0. Then we train the model on the train set. The API for logistic regression is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0) # Define the logistic regression model
model.fit(X_train, Y_train) # Train the model on the train set


# ### 5. Evaluate the trained logistic regression model on the test set

# In[ ]:


from sklearn.metrics import accuracy_score

Y_pred = model.predict(X_test) # Make prediction on the test set
acc = accuracy_score(Y_test, Y_pred) # Compute accuracy score on test set
print('Accuracy on test set:', acc)


# ### 6. Visualize the decision boundary
# 
# To visualize the decision boundary of a model on 2d inputs, we need to do the following 3 steps:
# * Create a grid of all points on the 2d input range.
# * Use the model to predict the probability of label 1 for each input.
# * Plot the contours of the predicted probabilities together with the data points.
# 
# Now we do **step 1**: creating a grid of all points on the 2d input range.

# In[ ]:


import numpy as np

X0 = X[:, 0]
X1 = X[:, 1]

# Find the range of the 2 dimensions that we will plot
X0_min, X0_max = X0.min()-1, X0.max()+1
X1_min, X1_max = X1.min()-1, X1.max()+1

n_steps = 100 # Number of steps on each axis

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(X0_min, X0_max, (X0_max-X0_min)/n_steps),
                     np.arange(X1_min, X1_max, (X1_max-X1_min)/n_steps))


# **Step 2:** we predict the model for each point on the meshgrid. Here we use the [predict_proba](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba) function to get the probability values. For each example, this function will return the probability for all labels. So the result will be an $n \times c$ matrix where $n$ is the number of examples and $c$ is the number of labels.

# In[ ]:


Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z1 = Z[:, 1] # Here we use the second column of the predictions, which corresponds to the label 1.
Z1 = Z1.reshape(xx.shape)


# **Step 3:** we plot the data and contour of the probability.

# In[ ]:


import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('x0')
plt.ylabel('x1')
plt.colorbar()
plt.show()


# 
# 
# 
# 
# 
# 
# 
# 
# 
