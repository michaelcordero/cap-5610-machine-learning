#!/usr/bin/env python
# coding: utf-8

# # CAP5610 - Numpy
# 
# Source: https://becominghuman.ai/an-essential-guide-to-numpy-for-machine-learning-in-python-5615e1758301
# 
# This demo only contains a subset of the tutorial above.

# ### 1. Import the library
# 
# First, we need to import the numpy library to use. Here `np` will be used as an abbreviation for `numpy`.

# In[ ]:


import numpy as np


# ### 2. Create a vector

# In[ ]:


# Create a row vector from a list
vector_row = np.array([1, 2, 3])
print(vector_row)
print(type(vector_row))

# Compare with the original list
print([1, 2, 3])
print(type([1, 2, 3]))


# In[ ]:


# Create a column vector from a list of lists
vector_column = np.array([[1],[2],[3]])
print(vector_column)
print(type(vector_column))

# Compare with the original list
print([[1],[2],[3]])
print(type([[1],[2],[3]]))


# ### 3. Create a matrix

# In[ ]:


# Create a matrix from a list of lists
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(matrix)
print(type(matrix))


# ### 4. Select elements

# In[ ]:


# Print 2nd element of vector_row
print(vector_row[1])


# In[ ]:


# Print 3rd element of vector_column
print(vector_column[2])


# In[ ]:


# Print element at 1st row, 3rd column of matrix
print(matrix[0, 2])


# In[ ]:


print(matrix)


# In[ ]:


# Print sub-matrix with 2nd row and first two columns
print(matrix[1:,:2])


# ### 5. Properties of a matrix

# In[ ]:


matrix = np.array([[1, 2, 3, 10],
                   [4, 5, 6, 11],
                   [7, 8, 9, 12]])

# Print number of rows and columns
print(matrix.shape)


# In[ ]:


# Print number of rows and columns separately
print(matrix.shape[0])
print(matrix.shape[1])


# In[ ]:


# Print number of elements
print(matrix.size)


# In[ ]:


# Print number of dimensions
print(matrix.ndim)


# ### 6. Modifying a matrix

# In[ ]:


matrix = np.array([[1, 2, 3, 10],
                   [4, 5, 6, 11],
                   [7, 8, 9, 12]])

# Modify one element
matrix[0, 3] = 20
print(matrix)


# In[ ]:


# Add a constant to all elements in the matrix
matrix += 5
print(matrix)


# In[ ]:


# Divide all elements in the matrix by a constant
matrix = matrix / 2
print(matrix)


# In[ ]:


# Modify one row of the matrix
matrix[1] = matrix[0]*2
print(matrix)


# ### 7. Find min and max elements

# In[ ]:


matrix = np.array([[1, 2, 3, 10],
                   [4, 5, 6, 11],
                   [7, 8, 9, 12]])

# Print max and min elements
print(np.max(matrix))
print(np.min(matrix))


# In[ ]:


# Print max element in each column
print(np.max(matrix, axis=0))


# In[ ]:


# Print min element in each row
print(np.min(matrix, axis=1))


# ### 8. Reshape and transpose

# In[ ]:


# Create a vector
vector = np.arange(12)
print(vector)


# In[ ]:


# Reshape the vector into a matrix
matrix = vector.reshape(4, 3)
print(matrix)


# In[ ]:


# Transpose the matrix
matrix = np.transpose(matrix)
print(matrix)


# In[ ]:


# Flatten the new matrix
print(matrix.flatten())


# ### 9. Matrix operations

# In[ ]:


m1 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])

m2 = np.array([[7, 8, 9],
               [4, 5, 6],
               [1, 2, 3]])


# In[ ]:


# Element-wise summation
print(m1+m2)


# In[ ]:


# Element-wise subtraction
print(m1-m2)


# In[ ]:


# Element-wise multiplication
print(m1*m2)


# In[ ]:


# Matrix multiplication
print(np.matmul(m1, m2))
print(m1 @ m2)


# ### 10. Random number generator
# * More examples and the complete APIs are available at: https://numpy.org/doc/stable/reference/random/generator.html

# In[ ]:


# Generate random integers uniformly from low to high-1
low = 1
high = 10

rng = np.random.default_rng()
print(rng.integers(low, high))
print(rng.integers(low, high))
print(rng.integers(low, high))


# In[ ]:


# Generate a matrix of random integers
print(rng.integers(low, high, size=(3, 5)))


# In[ ]:


# Generate random samples from a normal distribution with given mean and standard deviation
m = 2
std = 0.5

print(rng.normal(m, std))
print(rng.normal(m, std))
print(rng.normal(m, std))


# In[ ]:


# Generate a matrix of random samples from the above normal distribution
print(rng.normal(m, std, size=(3, 2)))


# ### 11. Exercises:
# * Complete the tutorial (some of them use old Numpy APIs): https://becominghuman.ai/an-essential-guide-to-numpy-for-machine-learning-in-python-5615e1758301
# * Try more exercises: https://www.machinelearningplus.com/python/101-numpy-exercises-python/
# * Practice, practice, practice. There are a lot of online exercises!
# * Use Google, stackoverflow.com, trial & error.
