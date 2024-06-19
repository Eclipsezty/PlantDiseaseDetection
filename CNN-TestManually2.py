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
# my_model = keras.models.load_model("F:/Study/FYP/training/models/Grape3-2__100.0.h5")
my_model = keras.models.load_model("F:/Study/FYP/training/models/Grape3-ResNet50_e60.h5")
#my_model2 = keras.models.load_model("F:/Study/FYP/training/models/Grape3-bi_e20_100.0.h5")




dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape3",  # load dataset from filename
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
    print(predictions[0])
    # find the max value of the possibility results, change it to percent,
    # then round the possibility(confidence) to a precision of 2 decimal digits.
    return predicted_class, confidence
    # return the predicted result class name and its possibility




def ergodic_pics(path):
    import os


    img_folder = path
    img_list = [os.path.join(nm) for nm in os.listdir(img_folder)]
    # img_list = [os.path.join(nm) for nm in os.listdir(img_folder) if nm[-3:] in ['JPG', 'png', 'gif']]
    print("total img number:",len(img_list))
    countBR = 0
    countE = 0
    countH = 0
    countL = 0
    icountBR = 0
    icountE = 0
    icountH = 0
    icountL = 0
    invalid = 0
    confidence_avg=0
    for i in img_list:
        imgpath = os.path.join(path, i)
        image = cv2.imread(imgpath)
        image = tf.image.resize(image, [200, 200])
        # plt.imshow(img.numpy().astype("uint8"))
        predicted_class, confidence = predict(my_model, image)
        if confidence>80 :
            confidence_avg += confidence
            if(predicted_class=="Grape___Black_rot"):
                countBR+=1
            elif(predicted_class=="Grape___Esca_(Black_Measles)"):
                countE+=1
            elif(predicted_class=="Grape___healthy"):
                countH+=1
            elif(predicted_class=="Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"):
                countL+=1
        # elif confidence>0:
        #     predicted_class, confidence = predict(my_model2, image)
        #     confidence_avg += confidence
        #     if (predicted_class == "Grape___Black_rot"):
        #         countBR += 1
        #     elif (predicted_class == "Grape___Esca_(Black_Measles)"):
        #         countE += 1
        #     elif (predicted_class == "Grape___healthy"):
        #         countH += 1
        #     elif (predicted_class == "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"):
        #         countL += 1
        else:
            invalid = invalid+1
            if (predicted_class == "Grape___Black_rot"):
                icountBR += 1
            elif (predicted_class == "Grape___Esca_(Black_Measles)"):
                icountE += 1
            elif (predicted_class == "Grape___healthy"):
                icountH += 1
            elif (predicted_class == "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"):
                icountL += 1
        print(f"Predicted: {predicted_class}. Confidence: {confidence}%")
    confidence_avg/=len(img_list)
    print(countBR,countE,countH,countL,"average confidence:",confidence_avg)
    resultm = [countBR,countE,countH,countL]
    invalidm= [icountBR,icountE,icountH,icountL]
    valid = len(img_list)-invalid
    acc = resultm[np.argmax(resultm)]/valid
    #print("class accuracy:",acc)
    print("invalid:",invalid)
    return acc,valid,resultm,invalid,invalidm


acc1, len1, resultm1,invalid1,invalidm1= ergodic_pics("F:/Study/FYP/training/Grape3-val/Grape___Black_rot")
print("**********************************************************************************************************************")
acc2,len2,resultm2,invalid2,invalidm2 =ergodic_pics("F:/Study/FYP/training/Grape3-val/Grape___Esca_(Black_Measles)")
print("**********************************************************************************************************************")
acc3,len3,resultm3,invalid3,invalidm3=ergodic_pics("F:/Study/FYP/training/Grape3-val/Grape___healthy")
print("**********************************************************************************************************************")
acc4,len4,resultm4,invalid4,invalidm4=ergodic_pics("F:/Study/FYP/training/Grape3-val/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)")

invalid_total = invalid1+invalid2+invalid3+invalid4
totalImg=invalid_total+len1+len2+len3+len4
acc_total = (acc1*len1+acc2*len2+acc3*len3+acc4*len4)/(len1+len2+len3+len4)
cm = [resultm1,resultm2,resultm3,resultm4]
im = [invalidm1,invalidm2,invalidm3,invalidm4]
dp = invalid_total/totalImg
print("total image number:",totalImg)
print("total acc: ",acc_total)
print("confusion matrix:", cm)
print("invalid matrix  :",im)
print("invalid number:",invalid_total)
print("discard proportion:",dp)
