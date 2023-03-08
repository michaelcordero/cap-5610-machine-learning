#!/usr/bin/env python
# coding: utf-8

# # Multi-layer perceptron
# 
# In this demo, we will train and test a multi-layer perceptron model on the MNIST handwritten digits dataset.

# ## 1. Load dataset
# 
# To load the original MNIST dataset, we can use the [fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml) method. Since this method will need to download the dataset and load it into memory, it will take a while.
# 
# Note that an MNIST image has shape 28 x 28. Here the images have already been flatten to a vector of size 784.

# In[ ]:


from sklearn.datasets import fetch_openml

X, Y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.  # Scale the pixel values to be in [0, 1]

print(X.shape)
print(Y.shape)


# ## 2. Split dataset into train/test sets
# 
# Traditionally, we use 10,000 examples for the test set.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10000, random_state=42)

print(X_train.shape)
print(X_test.shape)


# ## 3. Show a training example and its label
# 
# We can reshape an example (with index **img_id**) in the training set into a 28 x 28 matrix and plot it using matplotlib.

# In[ ]:


import matplotlib.pyplot as plt

img_id = 26
image = X_train[img_id].reshape((28, 28))
label = Y_train[img_id]

plt.imshow(image)
plt.show()
print(label)


# ### 4. Define and train an MLP classifier
# 
# We will define and train an MLP classifier with one hidden layer that contains 50 neurons and uses ReLU activation. This classifier will be trained with SGD for 30 epochs. We also set 'verbose=True' to track the training progress.
# 
# The API for this classifier is at: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# In[ ]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=30, solver='sgd', verbose=True)
mlp.fit(X_train, Y_train)


# ## 5. Show a test example and its predicted label

# In[ ]:


img_id = 30
image = X_test[img_id]

# Convert the image into 28 x 28 matrix to plot
plt.imshow(image.reshape((28, 28)))
plt.show()

print(image.shape)

# Convert the image vector into a matrix 1 x 784 before prediction
predicted_label = mlp.predict(image.reshape((1, 784)))
print(predicted_label)


# ### 6. Evaluate the trained MLP on the test set

# In[ ]:


from sklearn.metrics import accuracy_score

Y_pred = mlp.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print('Accuracy on test set:', acc)

