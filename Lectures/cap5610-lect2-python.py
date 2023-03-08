#!/usr/bin/env python
# coding: utf-8

# # CAP5610 - Python
# 
# * Here we are using a Jupyter notebook to code with Python 3. 
# * A Jupyter notebook (.ipynb) file can contain text cells (such as this cell) as well as code cells.
# * You can format text cells with Markdown.

# ## 1. Arithmetic operations
# 
# * We will create some variables and do arithmetic operations. 
# * Note different ways to print the results and how we add comments to the code.

# In[ ]:


x = 10
y = 4

z = x + y
print('Addition:', z)

z = x - y
print('Subtraction: %i' %z) # We can also print like C

print('Multiplication:', x * y)
print('Division:', x / y)
print('Modulus:', x % y)
print('Exponentiation:', x ** y)


# ## 2. Lists
# * A commonly used data structure
# * Can store different objects, even if they have different types
# * Complete API and examples for Python list: https://docs.python.org/3/tutorial/datastructures.html

# In[ ]:


# Create a new list with 7 elements
lst = [1, 3, 6, 4, 5, 3, 6]

print(lst) # Print the list's content
print(type(lst)) # Print the type of a variable


# In[ ]:


print(lst[2]) # Access the element at an index
print(lst[2:5]) # Access a sublist, this is called 'slicing'


# In[ ]:


# Add one element to the end of the list
lst.append(1)
print(lst)


# In[ ]:


# Remove first appearance of a value from list. How do we remove by index?
lst.remove(3)
print(lst)


# In[ ]:


# Add many elements to the list
lst.extend([7, 8])
print(lst)


# ## 3. Sets
# * Unordered collection with no duplicate elements
# * API and examples for Sets: https://docs.python.org/3/tutorial/datastructures.html#sets

# In[ ]:


# Create a set from the previous list
s = set(lst)
print(s)
print(type(s))


# In[ ]:


# Add elements to set
s = s.union({7, 9})
print(s)


# In[ ]:


# Set difference
s = s.difference({6, 7, 10})
print(s)


# ## 4. Tuples
# * Tuples are similar to lists, but immutable.
# * API and examples for Tuples: https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences

# In[ ]:


# Create a new tuple with 4 elements
t = (1, 2, 4, 5)
print(t)
print(type(t))
print(t[2])
print(t[2:4])


# In[ ]:


# Add elements to a tuple
t1 = t + (3, 2) # This operator concatenates two tuples together
print(t1)

# Remove 4 from the tuple
t2 = t1[:2] + t1[3:]
print(t2)


# ## 5. Dictionary
# * Data structure that contains unordered mappings of unique keys to values
# * API and examples for dictionary: https://docs.python.org/3/tutorial/datastructures.html#dictionaries

# In[ ]:


# Create new dictionary with 3 key-value pairs
d = {'key1': 'value1', 'key2': 3, 'key3': [2, 4]}
print(d)
print(d['key3'])


# In[ ]:


# Add element to dictionary
d['key4'] = 'new element'
print(d)


# In[ ]:


# Delete element from dictionary
del d['key1']
print(d)


# ## 6. Strings
# * You can access substring by slicing
# * Negative index can be used to count from the end of the string (or list)
# * API and examples for strings: https://docs.python.org/3/library/string.html

# In[ ]:


# Create a new string
s = 'I am taking CAP5610!'
print(s)


# In[ ]:


# Print some substrings, note how negative index can be used to count from the end of the string
print("These substrings are the same: %s, %s" %(s[12:19], s[-8:-1]))
print("Some other substrings: %s, %s" %(s[:4], s[12:]))


# In[ ]:


""" We can split a string into tokens """
tokens = s.split()
print("Tokens:", tokens)


# ## 7. Code flow
# * Python uses **indentation** to control blocks of code. 
# * Codes in the same block must have the same indentation. 
# * It's important to be consistent with *tab* and *space* here. 
# * If you use other editors, it would be best to convert all tabs to spaces to avoid errors. For Kaggle, it should already be done for you.

# In[ ]:


# Use for loop to add elements one-by-one to a list
lst = []
for i in range(10):
    lst.append(i)

print(lst)


# In[ ]:


# Use while loop to print numbers smaller than 5
i = 0
while i < 5:
    print(i)
    i += 1


# In[ ]:


# A very handy one-line loop
lst = [i*2 for i in range(3, 8)]
print(lst)


# In[ ]:


# Another one-line loop
lst = [i*j for i in range(4) for j in range(1, 3)]
print(lst)


# ## 8. Functions
# * Functions in Python can be define with or without type information. 
# * They can also take other functions as inputs (higher-order functions).

# In[ ]:


# Define a new function f
def f(x, y):
    return x ** y

print(f(2, 3))


# In[ ]:


# Apply f to each element of a list
l = [2, 4, 7, 3]
l1 = [f(x, 2) for x in l]
print(l1)


# In[ ]:


# Function can also take another function as argument
def g(func, z):
    x = func(z, 2)
    return x + 1

y = g(f, 3)

print(y)


# In[ ]:


# Some arguments can take default values (only the last arguments)
def h(x, y=3, z=5):
    return [x, y, z]

print(h(3, 4, 5))
print(h(3))


# ## 9. Object-oriented Python

# In[ ]:


# Define a class
class Multiply:
    """ A class that will multiply an input by a factor """

    # Constructor
    def __init__(self, factor=2):
        self.factor = factor

    # Multiply method
    def mul(self, x):
        return x * self.factor


# In[ ]:


# Create an object and call the mul method
m3 = Multiply(3)
print(m3.mul(10))


# In[ ]:


m2 = Multiply()
print(m2.mul(10))


# In[ ]:


# Define an inherited class
class MultiplyBy4(Multiply):
    """ A class that will multiply an input by 4 """

    # Override the constructor
    def __init__(self):
        super().__init__(4) # Call the constructor of the parent class

# Create an object and call the mul method
m4 = MultiplyBy4()
print(m4.mul(7))


# ## 10. Exercises
# * Complete the short Python tutorial: https://medium.com/codex/python-a-short-tutorial-f1d9c65b6e87
# * Complete the long Python tutorial: https://www.w3schools.com/PYTHON/default.asp
