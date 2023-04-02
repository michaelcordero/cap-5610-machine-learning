#!/usr/bin/env python
# coding: utf-8

# # CAP5610 - Assignment 1 - Programming question (11%)
# 
# In this homework, we will train and test several linear regression models on the diabetes dataset. Details of the dataset can be found here: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset. **After completing this assignment, download your notebook and upload it to Canvas.**
# 
# First, run the following code snippet to load the dataset. After executing the code, the inputs and corresponding outputs will be loaded to the variables X and Y respectively. We also print out the shapes of X and Y for a sanity check.

# In[1]:


from sklearn.datasets import load_diabetes

X, Y = load_diabetes(return_X_y=True)

print(X.shape, Y.shape)


# ## 1. Split data (1%)
# 
# Write code to randomly split the data into a train set (70% total data) and test set (30% total data). Store your splits into 4 variables X_train, X_test, Y_train, Y_test that correspond to the train inputs, test inputs, train labels, and test labels respectively.

# In[3]:


# Write your code here
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# ## 2. Train and evaluate a linear regression model (4%)
# 
# Write code to train and evaluate a linear regression model using Sklearn. The API for the model is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html. 
# 
# Your code should do the following steps:
# 1. Define a linear regression model with default parameters.
# 2. Fit the model using the train set.
# 3. Use the model to make predictions on the test set.
# 4. Compute and print out the mean squared error on the test set.

# In[6]:


# Write your code here
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
Y_prediction = lr_model.predict(X_test)
mse = mean_squared_error(Y_test, Y_prediction)
print(f'Mean Squared Error for LR Model: {mse}')


# ## 3. Train and evaluate a Ridge regression model (2%)
# 
# Write code to train and evaluate a Ridge regression model using Sklearn. The API for the model is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html.
# 
# Your code should follow the same 4 steps as in Question 2 above. Use the default regularization **alpha=1.0** for the model.

# In[35]:


# Write your code here
from sklearn.linear_model import Ridge
rr_model = Ridge(alpha=1.0)
rr_model.fit(X_train, Y_train)
rrY_prediction = rr_model.predict(X_test)
rr_mse = mean_squared_error(Y_test, rrY_prediction)
print(f'MSE for RR Model: {rr_mse}')


# ## 4. Train and evaluate a Lasso regression model (2%)
# 
# Write code to train and evaluate a Lasso regression model using Sklearn. The API for the model is at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html.
# 
# Your code should follow the same 4 steps as in Question 2 above. Use the default regularization **alpha=1.0** for the model.

# In[36]:


# Write your code here
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, Y_train)
lmY_prediction = lasso_model.predict(X_test)
lm_mse = mean_squared_error(Y_test, lmY_prediction)
print(f'MSE for Lasso Model: {lm_mse}')


# ## 5. Test various alpha values for Ridge and Lasso model (2%)
# 
# Re-run question 3 and 4 with various values of **alpha** according to the table below. Fill the table with the test MSE of these models. Which model (for example, Ridge with alpha=0.6, etc.) performs best on the test set? 
# 
# You can edit the table and add your answer directly into this cell by double-clicking on it. **If you modify your code in question 3 and 4 to test alpha values, remember to change it back to the default alpha=1.0 before submitting your code.**
# 
# | model/alpha | 0.1     | 0.2     |  0.4    | 0.6     | 0.8     | 1.0       | 10.0     | 100.0   |
# | ---         |---------|---------|---------|---------|---------|-----------|----------|---------|
# |  Ridge      | 2805.40 | 2813.15 | 2870.31 | 2948.20 | 3030.97 | 3312.96   | 4580.10  | 5318.57 |
# |  Lasso      | 2775.16 | 2806.10 | 2889.19 | 3003.14 | 3195.80 | 3444.67   | 5432.88  | 5432.88 |
# 
# **Your answer:** ...

#  **Lasso Model with alpha=0.1 has the lowest MSE score, therefore it performs the best on the test set.**
