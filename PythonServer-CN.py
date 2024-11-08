import os
import socket
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import os

# from PythonServer import TFimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试其他 backend 如 'Agg', 'Qt5Agg', 'TkAgg' 等
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

address = ('192.168.0.102', 6666)

# Load the models
my_model = keras.models.load_model("F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5")
my_model2 = keras.models.load_model("F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5")


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    class_names = ['Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast']
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


def bi_model_predict(my_model, my_model2, img):
    predicted_class, confidence = predict(my_model, img)
    if confidence >= 60:
        return predicted_class, confidence
    else:
        return predict(my_model2, img)


def translate_CN(eng):
    if eng == "Rice___Brown_Spot":
        return "褐斑病"
    elif eng == "Rice___Healthy":
        return "健康"
    elif eng == "Rice___Leaf_Blast":
        return "叶瘟"
    elif eng == "Rice___Neck_Blast":
        return "穗颈瘟"
    else:
        return "未知疾病"


while True:
    # Create socket
    tcpServe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServe.bind(address)
    tcpServe.listen(5)
    print("服务器启动，等待连接...")

    # Wait for a client connection
    tcpClient, addr = tcpServe.accept()
    print('客户端IP地址:', addr)
    base64_data = ''

    while True:
        data = tcpClient.recv(1024)
        base64_data += str(data, 'utf-8').strip()
        if not data:
            break

    # Split data into image data, GPS coordinates, and phone number
    try:
        image_data_str, gps_data_str, phone_number = base64_data.split('|')
        latitude, longitude = gps_data_str.split(',')
        latitude = float(latitude)
        longitude = float(longitude)
        print(f"Received GPS location - Latitude: {latitude}, Longitude: {longitude}")
        print(f"Received Phone Number: {phone_number}")
    except ValueError:
        print("Error parsing data")
        latitude, longitude, phone_number = None, None, "Unknown"

    # Decode the image
    img = base64.b64decode(image_data_str)
    image_data = BytesIO(img)
    im = Image.open(image_data)

    # Save image
    imageNum = max([int((i.split(".")[0]).split("Data")[1]) for i in os.listdir("F:/Study/FYP/training/Image")]) + 1
    img_path = f"F:/Study/FYP/training/Image/ImageData{imageNum}.jpg"
    im.save(img_path)

    TFimg = tf.image.resize(cv2.imread(img_path), [200, 200])

    # Run the prediction and store results
    predicted_class, confidence = bi_model_predict(my_model, my_model2, TFimg)
    result = f"检测结果：{translate_CN(predicted_class)}  可信度：{confidence}%"
    print(result)
    try:
        message = result
    except:
        message = "0"
    tcpClient.send((message + "\n").encode())
    print("P Complete")

    # Optionally, save the results including GPS data and phone number
    if latitude is not None and longitude is not None:
        with open("detection_results.csv", "a", encoding="utf-8") as f:
            f.write(f"{img_path},{translate_CN(predicted_class)},{confidence},{latitude},{longitude},{phone_number}\n")

    tcpClient.close()
    tcpServe.close()
