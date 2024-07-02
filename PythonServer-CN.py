import os
import socket
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import os
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

address = ('192.168.206.217', 6666)

my_model = keras.models.load_model("F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5")
my_model2 = keras.models.load_model("F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5")

# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     r"F:\Research\CAMIC\New CAMIC\Train"  # load dataset from filename to get class names
# )

def predict(model, img):

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    # increasing dimension, then it participate in tensor model calculation
    predictions = model.predict(img_array)
    class_names = ['Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast']

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
    if confidence >= 60:
        return predicted_class, confidence
    else:
        predicted_class, confidence = predict(model2, img)
        return predicted_class, confidence

def translate_CN(eng):
    if(eng == "Rice___Brown_Spot"):
        return "褐斑病"
    elif(eng == "Rice___Healthy"):
        return "健康"
    elif(eng == "Rice___Leaf_Blast"):
        return "叶瘟"
    elif(eng == "Rice___Neck_Blast"):
        return "穗颈瘟"
    else:return "未知疾病"

while True:
    # 1. create socket
    tcpServe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    base64_data = ''
    # 2. bind socket
    tcpServe.bind(address)
    # 3. receive connection request
    tcpServe.listen(5)
    print("服务器启动，等待连接...")

    # 4. wait client
    tcpClient, addr = tcpServe.accept()
    print('客户端IP地址:', addr)
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
    # plt.figure(figsize=(12, 12))
    # plt.imshow(TFimg.astype("uint8"))
    # plt.title(f"Predicted: {predicted_class}.\n Confidence: {confidence}%", fontsize=36)
    # plt.axis("off")
    # plt.show()
    if(confidence<60):
        result = "无法识别，请重新拍照"
        print("可信度低：",confidence)
    else:
        result = f"检测结果：{translate_CN(predicted_class)}  可信度：{confidence}%"
    print(result)
    try:
        message = result
    except:
        message = "0"
    tcpClient.send((message + "\n").encode())
    print("识别完成")

    tcpClient.close()
    tcpServe.close()

