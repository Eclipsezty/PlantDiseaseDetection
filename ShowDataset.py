import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import tensorflow as tf

A1 = np.arange(1, 6)  # x ∈ [1,5]
A2 = np.arange(1, 11) # x ∈ [1,10]
B = A1 ** 2
C = A1 ** 3
D = A2 ** 2

# create figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

# plot data
ax1.bar(A1, B)
ax2.scatter(A1, C)
ax3.plot(A2, D)

plt.show()






dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "F:/Study/FYP/training/plantvillage dataset/color",  # load dataset from filename
)
class_names = dataset.class_names
data = tf.data.experimental.cardinality(dataset)

class_count = len(class_names)
print(class_count)
lenth=[]
for i in range(class_count):
    classname = "F:/Study/FYP/training/plantvillage dataset/color/"+class_names[i]
    files = os.listdir(classname)
    num = len(files)
    #print(num)
    lenth.append(num)


x=class_names
y=lenth
plt.figure(figsize=(8,6))

#plt.barh(lenth, tick_label=class_names)
plt.barh(x,y)
#plt.xticks(x, x, rotation=90)
plt.ylabel('Categories')
plt.xlabel('Number of Images')
plt.title('PlantVillage Dataset')
tight_layout = True
plt.tight_layout()
plt.show()