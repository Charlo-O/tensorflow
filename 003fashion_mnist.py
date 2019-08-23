#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf 
print (tf.__version__)


# In[6]:


mnist=tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
import matplotlib.pyplot as plt
plt.imshow(training_images[10086])
print(training_labels[10086])
print(training_images[10086])
training_images=training_images/ 255.0
test_images=test_images/255.0


# In[10]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128,activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10,activation=tf.nn.softmax)])


# In[14]:


model.compile(optimizer = tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')
model.fit(training_images,training_labels,epochs=5)


# In[16]:


model.evaluate(test_images,test_labels)


# In[ ]:




