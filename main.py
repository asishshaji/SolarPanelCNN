#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


from utils.elpv_reader import load_dataset
images, proba, types = load_dataset()


# In[3]:


import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
height = 200


# In[4]:


images = images /255.
images = images.reshape(2624,height,height,1)


# In[5]:


model = models.Sequential()

model.add(layers.Conv2D(64, (7, 7), input_shape=(height, height,1),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5),activation='relu'))
      
model.add(layers.Conv2D(32, (5, 5),activation='relu'))

          
          
#FCC
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
          
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
model.summary()


# In[ ]:


model.fit(images[:2000],proba[:2000],epochs = 5)
model.evaluate(images[2000:],proba[2000:])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




