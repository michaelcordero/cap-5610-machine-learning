#!/usr/bin/env python
# coding: utf-8

# # Multi-layer perceptrons with Keras
# 
# In this demo, we will train and test a multi-layer perceptron model on the MNIST handwritten digits dataset using Keras.

# ## 1. Load dataset
# * Like sklearn, Keras also provides an API to download and load the MNIST dataset. 
# * The following code snippet will download the data, load it into memory, and convert pixel values to [0, 1].
# * The API for the method is at: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
# * Note that here the images are not flatten.

# In[ ]:


from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# ## 2. Show a few training examples and its label
# * We show a training example based on index **img_id** as in the previous demo.

# In[ ]:


import matplotlib.pyplot as plt

img_id = 59
image = X_train[img_id]
label = Y_train[img_id]

plt.imshow(image)
plt.show()
print(label)


# ## 3. Flatten the inputs into vectors
# * The images are not flattened, so we flatten them into vectors to train with an MLP model.

# In[ ]:


X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(X_train.shape, X_test.shape)


# ## 4. Convert label vectors into one-hot encodings
# * When using Keras for classification, the labels have to be converted into one-hot encoding vectors.
# * We can do this using the [to_categorial](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical) method.

# In[ ]:


from tensorflow.keras.utils import to_categorical

num_classes = 10
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

print(Y_train.shape, Y_test.shape)


# ## 5. Define the MLP model
# * We can define an MLP model using a [Sequential](https://keras.io/api/models/sequential/) model and the [Dense](https://keras.io/api/layers/core_layers/dense/) layers. 
# * In most cases, we will define a model as a Sequential model, and then add layers to it one-by-one.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))


# ## 6. Compile the model
# * Before training a Keras model, we need to compile it to set up all the options for training, such as loss function, optimizer, and evaluation metrics. 
# * Here we will use cross entropy loss and the SGD optimizer. Our evaluation metric will be accuracy.

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# ## 7. Train the model
# * Now we can train the model using the `fit(...)` method. 
# * We can specify the number of epochs and batch size for training when calling fit().

# In[ ]:


model.fit(X_train, Y_train, epochs=30, batch_size=128)


# ## 8. Evaluate the trained model on test set
# 
# Finally, we can compute the model accuracy on the test set.

# In[ ]:


_, accuracy = model.evaluate(X_test, Y_test)

print(accuracy)

