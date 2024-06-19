import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#% matplotlib inline
import numpy as np
import glob
import os

train_image_path = glob.glob("D:/Grade4/FYP/Working/training/Grape/*/*.jpg")
# int(p.split('training_set/training_set/')[1].split('/')[0] == 'cats')
train_image_label = [int(p.split('training_set/training_set/')[1].split('/')[0] == 'cats') for p in train_image_path]


# 通过路径读取图片
def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    # 解码
    image = tf.image.decode_jpeg(image, channels=3)
    # 将图像处理成相同大小
    image = tf.image.resize(image, [256, 256])

    # 数据增强
    image = tf.image.random_crop(image, [256, 256, 3])  # 随机裁剪
    image = tf.image.random_flip_left_right(image)  # 随机左右翻转
    image = tf.image.random_flip_up_down(image)  # 随机上下翻转
    image = tf.image.random_brightness(image, 0.5)  # 随机调整亮度
    image = tf.image.random_contrast(image, 0, 1)  # 随机对比度

    # 改变数据类型
    image = tf.cast(image, tf.float32)
    # 归一化
    image = image / 255

    # 将label处理成二维的: [1,2,3] ---> [[1],[2],[3]]
    label = tf.reshape(label, [1])
    return image, label


train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
BATCH_SIZE = 32
train_count = len(train_image_path)

img, lab = next(iter(train_image_ds))
train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
train_image_ds = train_image_ds.prefetch(AUTOTUNE)

test_image_path = glob.glob("../input/cat-and-dog/test_set/test_set/*/*.jpg")
test_image_label = [int(p.split('test_set/test_set/')[1].split('/')[0] == 'cats') for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
test_image_ds = test_image_ds.batch(BATCH_SIZE)
test_image_ds = test_image_ds.prefetch(AUTOTUNE)

model = keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])

ls = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.Accuracy()

epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.Accuracy()


def train_step(model, images, labels):
    with tf.GradientTape() as t:  # 记录梯度计算过程(上下文管理器)
        pred = model(images)
        # from_logits=True输出的预测结果未激活   第一个参数为正确值，第二个参数为预测值
        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    grads = t.gradient(loss_step, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuracy(labels, tf.cast(pred > 0, tf.int32))


def test_step(model, images, labels):
    pred = model(images, training=False)  # 或model.predict(images)
    # from_logits=True输出的预测结果未激活   第一个参数为正确值，第二个参数为预测值
    loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    epoch_loss_avg_test(loss_step)
    test_accuracy(labels, tf.cast(pred > 0, tf.int32))


train_loss_result = []
train_acc_result = []

test_loss_result = []
test_acc_result = []

num_epochs = 30

for epoch in range(num_epochs):
    for imgs_, labels_ in train_image_ds:
        train_step(model, imgs_, labels_)
        print('.', end='')
    print()

    train_loss_result.append(epoch_loss_avg.result())
    train_acc_result.append(train_accuracy.result())

    for imgs_, labels_ in test_image_ds:
        test_step(model, imgs_, labels_)

    test_loss_result.append(epoch_loss_avg_test.result())
    test_acc_result.append(test_accuracy.result())

    print('Epoch:{}: loss:{:.3f}, accuracy:{:.3f}, test_loss:{:.3f},test_accuracy:{:.3f}'.format(
        epoch + 1,
        epoch_loss_avg.result(),
        train_accuracy.result(),
        epoch_loss_avg_test.result(),
        test_accuracy.result()
    ))
    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()

    epoch_loss_avg_test.reset_states()
    test_accuracy.reset_states()