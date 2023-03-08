#!/usr/bin/env python
# coding: utf-8

# # CAP5610 - Linear regression
# * In this demo, we will train and test linear regression models on a toy linear dataset: $Y = 10*X + \epsilon$, where $\epsilon$ is Gaussian noise with zero mean and standard deviation 0.8. 
# * You can also use sklearn's API to generate different datasets: https://scikit-learn.org/stable/modules/classes.html#samples-generator
# * Here we will generate the dataset manually.

# ### 1. Generate the toy dataset
# * We generate a dataset with 200 samples.
# * An input $x$ contains 1 feature uniformly drawn from (-1, 1).
# * The corresponding output is $10*x + \epsilon$, where $\epsilon \sim N(0, 0.8^2)$.
# * We store all the inputs into a matrix $X$ and store the corresponding labels into a matrix $Y$.

# In[ ]:


import numpy as np

# Create a random number generator with fixed seed
rng = np.random.default_rng(12345)
n_samples = 200

# We randomly draw X, noise, and compute Y
X = rng.uniform(-1, 1, size=(n_samples, 1))
noise = rng.normal(scale=0.8, size=(n_samples, 1))
Y = 10*X + noise


# In[ ]:


# Print some information of the dataset
print(X.shape)
print(Y.shape)

print(X[:5])
print(Y[:5])


# ### 2. Visualize the dataset
# * We can visualize the dataset using the matplotlib library.

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X, Y) # Plot all data points with default color
plt.show() # Show the plot


# ### 3. Create train/test sets
# * We will randomly split the dataset into a train and a test set.
# * Sklearn already has a function to do that. The API for this function is at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
# * Here the test set will be 40% of the whole dataset.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=10)

print(X_train.shape, X_test.shape)


# ### 4. Visualize the train/test sets
# * We now visualize the train/test sets by plotting them with different colors

# In[ ]:


import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color='black') # Plot train points with black color
plt.scatter(X_test, Y_test, color='red') # Plot test points with red color
plt.show() # Show the plot


# ### 5. Train a linear regression model on the train set
# * Now we create and train a linear regression model on the train set.
# * We can create a LinearRegression object for the model.
# * Then we call the function fit() to train the model on a dataset.
# * The full API for the model is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# In[ ]:


from sklearn.linear_model import LinearRegression

# Create a linear regression model with default parameters
model = LinearRegression()

# Train the model on the train set
model.fit(X_train, Y_train)


# ### 6. Evaluate the trained linear regression model on the test set
# * We evaluate the trained model on the test set by computing the mean squared error.
# * Sklearn also has a function to compute the mean squared error and other metrics.
# * Check the API at: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# In[ ]:


from sklearn.metrics import mean_squared_error

# We first use the model to predict the labels on the test set
Y_pred = model.predict(X_test)

# Next we compute the mean squared error between true and predicted labels
mse = mean_squared_error(Y_test, Y_pred)

# Finally we print out the mse
print(mse)


# ### 7. Visualize the trained model
# * We can visualize the trained model by plotting the predictions for examples on the whole input range (-1 ,1).

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Generate 150 inputs evenly spaced in the input range and reshape them into a 150 x 1 matrix (each row is an example)
xx = np.linspace(min(X), max(X), 150).reshape(-1, 1)

# Make prediction on these inputs
yy = model.predict(xx)

# Plot a line connecting these points
plt.plot(xx, yy, color='blue', label='model', linewidth=2)

# Plot train points with black color
plt.scatter(X_train, Y_train, color='black', label='train')

# Plot test points with red color
plt.scatter(X_test, Y_test, color='red', label='test')

# Add the legend
plt.legend()

# Show the plot
plt.show()


# ### 8. Ridge regression
# 
# We can also train and evaluate a Ridge regression model using similar codes. The only difference is that we replace **model = LinearRegression()** in step 5 above by **model = Ridge(alpha=...)** after importing the Ridge class. The API for Ridge regression is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

model = Ridge(alpha=0.5) # Create a Ridge regression model with alpha=0.5
model.fit(X_train, Y_train) # Train the model on the train set
Y_pred = model.predict(X_test) # Use the model to predict the labels on the test set
print(mean_squared_error(Y_test, Y_pred)) # Compute and print out the MSE on the test set


# ### 9. Lasso regression
# 
# We can also train and evaluate a Lasso regression model using similar codes. The only difference is that we replace **model = LinearRegression()** in step 5 above by **model = Lasso(alpha=...)** after importing the Lasso class. The API for Lasso is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html.

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

model = Lasso(alpha=0.5) # Create a Lasso regression model with alpha=0.5
model.fit(X_train, Y_train) # Train the model on the train set
Y_pred = model.predict(X_test) # Use the model to predict the labels on the test set
print(mean_squared_error(Y_test, Y_pred)) # Compute and print out the MSE on the test set


# In[ ]:




