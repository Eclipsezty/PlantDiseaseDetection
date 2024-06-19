import os
import socket
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

address = ('192.168.250.217', 6666)

my_model = keras.models.load_model("F:/Study/FYP/training/models/Grape3-Xception_e30_9637.h5")
my_model2 = keras.models.load_model("F:/Study/FYP/training/models/Grape3-Eff_e50_9626.h5")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape3",  # load dataset from filename to get class names
)

def predict(model, img):

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    # increasing dimension, then it participate in tensor model calculation
    predictions = model.predict(img_array)
    class_names = dataset.class_names

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    # find the max value of the possibility results, change it to percent,
    # then round the possibility(confidence) to a precision of 2 decimal digits.
    return predicted_class, confidence
    # return the predicted result class name and its possibility

def bi_model_predict(my_model,my_model2, img):
    model1 = my_model
    model2 = my_model2
    predicted_class, confidence = predict(model1,img)
    if confidence >= 80:
        return predicted_class, confidence
    else:
        predicted_class, confidence = predict(model2, img)
        return predicted_class, confidence


while True:
    # 1. create socket
    tcpServe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    base64_data = ''
    # 2. bind socket
    tcpServe.bind(address)
    # 3. receive connection request
    tcpServe.listen(5)
    print("Server running, waiting for connection...")

    # 4. wait client
    tcpClient, addr = tcpServe.accept()
    print('Client address:', addr)
    while True:
        data = tcpClient.recv(1024)
        base64_data += str(data, 'utf-8').strip()
        if not data:
            break

    img = base64.b64decode(base64_data)
    image_data = BytesIO(img)
    im = Image.open(image_data)

    imageNum = max([int((i.split(".")[0]).split("Data")[1]) for i in os.listdir("F:/Study/FYP/training/Image")]) + 1
    img_path = f"F:/Study/FYP/training/Image/ImageData{imageNum}.jpg"
    im.save(f"F:/Study/FYP/training/Image/ImageData{imageNum}.jpg")
    TFimg = cv2.imread(img_path)
    TFimg = tf.image.resize(TFimg,[200,200])

    predicted_class, confidence = bi_model_predict(my_model,my_model2,TFimg)
    plt.figure(figsize=(12, 12))
    plt.imshow(TFimg.astype("uint8"))
    plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%", fontsize=36)
    plt.axis("off")
    plt.show()
    if(confidence<60):
        result = "Cannot Recognize"
    else:
        result = f"Predicted: {predicted_class}.  Confidence: {confidence}%"
    print(result)
    try:
        message = result
    except:
        message = "0"
    tcpClient.send((message + "\n").encode())
    print("P Complete")

    tcpClient.close()
    tcpServe.close()