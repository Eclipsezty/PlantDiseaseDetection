import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# my_model = keras.models.load_model("F:/Study/FYP/training/models/plant disease__81.37.h5")
# my_model = keras.models.load_model("F:/Study/FYP/training/models/plant disease__99.50.h5")
my_model = keras.models.load_model("F:/Study/FYP/training/models/plant disease__99.01.h5")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape",  # load dataset from filename
)

def predict(model, img):

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    # transform the images to numpy array
    img_array = tf.expand_dims(img_array, 0)
    # increasing dimension, then it an participate in tensor model calculation
    predictions = model.predict(img_array)
    class_names = dataset.class_names

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    # find the max value of the possibility results, change it to percent,
    # then round the possibility(confidence) to a precision of 2 decimal digits.
    return predicted_class, confidence
    # return the predicted result class name and its possibility

img_path = 'F:/Study/FYP/training/Grape2/Grape___Black_rot/3432b9e9-a51a-4ce2-973e-b91e311ae6ef___FAM_B.Rot 3362.JPG'
img_path2 = 'F:/Study/FYP/training/Grape2/Grape___Esca_(Black_Measles)/0ad02171-f9d0-4d0f-bdbd-36ac7674fafc___FAM_B.Msls 4356.JPG'
img_path3 = 'F:/Study/FYP/training/Grape2/Grape___healthy/0ac4ff49-7fbf-4644-98a4-4dc596e2fa87___Mt.N.V_HL 9004.JPG' #healthy
# img_path = 'F:/Study/FYP/training/Grape2/Grape___Black_rot/3588f6c1-e117-4f85-b9e5-b2082d6dcf52___FAM_B.Rot 3419.JPG'
# img = tf.io.read_file(img_path)  # bytes
# img = tf.image.decode_jpeg(image_raw)
img = cv2.imread(img_path3)
img = tf.image.resize(img,[200,200])
plt.figure(figsize=(12, 12))
plt.imshow(img.astype("uint8"))
# plt.imshow(img.numpy().astype("uint8"))
predicted_class, confidence = predict(my_model, img)
# predicted_class, confidence = predict(my_model, img.numpy())
        # get data and print the result in image
#actual_class = class_names[labels[i]]
plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%",fontsize=36)
#plt.title(f"Actual:{actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
plt.axis("off")
plt.show()