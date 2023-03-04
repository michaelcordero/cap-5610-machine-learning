#!/usr/bin/env python
# coding: utf-8

# **<h2>Image classifier with Keras</h2>**

# **<h3>1. Import modules</h3>**
# * *Image* is used later to load the images.
# * *Sequential* allows you to add layers to the model in sequential order.
# * *Dense*, *Conv2D*, *MaxPooling2D*, *Dropout*, and *Flatten* are layers used to add to the model later.
# * *Adam* is an optimizer that uses learning rates for each parameter during training.
# * *LabelEncoder* is used to encode categorical labels as integer values.

# In[2]:


import os
import csv
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# **<h3>2. Get paths for datasets and images</h3>**

# In[4]:


# Set the directories for the image dataset and the training dataset
print(os.getcwd())
# os.chdir('/Users/michaelcordero/PycharmProjects/cap-5610-machine-learning/Final')
# print(os.getcwd())
image_dir = 'input/fiu-cap5610-spring-2023/images'
train_data_file = 'input/fiu-cap5610-spring-2023/train.csv'
test_data_file = 'input/fiu-cap5610-spring-2023/test.csv'


# **<h3>3. Set image dimensions and number of possible labels</h3>**

# In[5]:


# Set the image dimensions and number of classes
img_rows, img_cols = 112, 224
num_classes = 20


# **<h3>4. Load datasets and extract data</h3>**

# In[6]:


# Load the training data
train_data = np.genfromtxt(train_data_file, delimiter=',', dtype=None, encoding=None, skip_header=1)
# Extract the image ids and labels from the training data
train_ids = [str(train_data[i][0]) for i in range(len(train_data))]
train_labels = [str(train_data[i][1]) for i in range(len(train_data))]

# Load the test data
test_data = np.genfromtxt(test_data_file, delimiter=',', dtype=None, encoding=None, skip_header=1)
# Extract the image ids from the test data
test_ids = [str(test_data[i]) for i in range(len(test_data))]


# **<h3>5. Encode labels from the training data and map to original strings</h3>**

# In[7]:


# Encode the labels with integers
le = LabelEncoder()
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Create a dictionary to map the encoded labels to their original strings
class_dict = dict(zip(le.transform(le.classes_), le.classes_))


# **<h3>6. Load images into numpy arrays</h3>**

# In[8]:


# Load the images and convert them to numpy arrays
def load_images(image_ids):
    images = []
    for image_id in image_ids:
        image_path = os.path.join(image_dir, str(image_id) + '.png')
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((img_cols, img_rows), resample=Resampling.BICUBIC)
            images.append(np.array(img))
    return np.array(images)


# **<h3>7. Define the MLP model</h3>**

# In[9]:


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) 


# **<h3>8. Compile the model</h3>**
# * Before training a Kearas model, we need to compile it to set up all the options for training, such as loss function, optimizer, and evaluation metrics.
# * Here we will use cross entropy loss and the Adam optimizer. Our evaluation metric will be accuracy.

# In[10]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# **<h3>9. Load training images and split them into train and validation sets</h3>**

# In[11]:


# Load the training images and split them into training and validation sets
X_train = load_images(train_ids)
X_train = X_train.astype('float32') / 255.0
y_train = np.eye(num_classes)[train_labels_encoded]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# **<h3>10. Train the model</h3>**

# In[12]:


# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=30, validation_data=(X_val, y_val))


# **<h3>11. Load test images</h3>**

# In[13]:


# Load the test images
X_test = load_images(test_ids)
X_test = X_test.astype('float32') / 255.0 


# **<h3>12. Predict test images</h3>**

# In[14]:


# Predict the labels for the test images
y_pred = model.predict(X_test)
y_pred_encoded = np.argmax(y_pred, axis=1)


# **<h3>13. Map labels and save the result on csv file</h3>**

# In[17]:


# Map the predicted encoded labels to their original strings
y_pred_classes = [class_dict[y] for y in y_pred_encoded]

# Create a list of dictionaries for the test predictions
predictions = []
for i, image_id in enumerate(test_ids):
    predictions.append({'id': image_id, 'class': y_pred_classes[i]})

# Write the predictions to a CSV file
with open('working/submission.csv', 'w', newline='') as f:
    fieldnames = ['id', 'class']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for prediction in predictions:
        writer.writerow(prediction)


# **<h3>*Optional: Print predicted labels for each test image</h3>**

# In[18]:



# Output the predicted labels for each test image
for i, image_id in enumerate(test_ids):
    print(f"Image {image_id}: predicted class is {y_pred_classes[i]}")


# In[ ]:




