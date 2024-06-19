# Packages to import
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# Defining Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32  # each batch will be used to training the model at one time. Why choose 32?
#  Small batch size training speed is slow, and it will aggravate overfitting.
#  Large batch size will increase training speed, fluctuate smaller but convergence slow, need more epochs.
#  The most common one in practical projects is the mini-batch, usually dozens or hundreds.
#  The GPU can play a better role in the power of 2 batch. Therefore, we choose 32 as our batch size.
CHANNELS = 3  # RGB three kinds of color
EPOCHS = 10  # one epoch means the model ergodic the whole dataset

print("Dataset Making")
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape3",  # load dataset from filename
    shuffle=True,  # disorder the sequence of the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # 256*256 pixels image
    batch_size=BATCH_SIZE  # each batch will be used to training the model at one time
)

# Printing Class Names
print("Class Names")
class_names = dataset.class_names
print(class_names)

print("Length of Dataset")
print(len(dataset))

for image_batch, label_batch in dataset.take(1):  # First Batch
    print(image_batch.shape)  # （batch size, x_pixel, y_pixel, channel)
    print(label_batch.numpy())  # print the label of images (whether infected diseases)

# Just to show Images in Batch1
for image_batch, label_batch in dataset.take(1):  # Batch 1
    for i in range(1):  # Print one image
        plt.imshow(image_batch[i].numpy().astype("uint8"))  # Tensor is converted to numpy(), which can be displayed
        # "uint8" is from 0-255
        plt.title(class_names[label_batch[i]])  # set image title which will be printed
        plt.axis("off")  # hide the axis
        plt.show()  # print the image


# Dataset Partitioning
def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    #  the 80% of dataset is used in training, 10% is used to validate whether the model is valid or overfitting
    #  using a buffer to shuffle, which size is 10000. Firstly set the first 10000 elements in the buffer, and pop, push randomly
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        # use random seed to store the random number, allow running code multiple times can get the same result

    train_size = int(train_split * ds_size)  # size of dataset * 0.8
    val_size = int(val_split * ds_size)  # size of dataset * 0.1

    train_ds = ds.take(train_size)
    # take(): get element from array, along axis
    val_ds = ds.skip(train_size).take(val_size)
    # skip(): skip the elements from training data, then get the validation data
    test_ds = ds.skip(train_size).skip(val_size)  # the remain elements  # or use .take(int(test_split * ds_size))

    return train_ds, val_ds, test_ds



train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)

print(len(train_ds))
print(len(val_ds))
print(len(test_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
#  Cache(): cache datasets into memory to speed up operation
#  shuffle(): disorder the sequence
#  The GPU/TPU is idle while the CPU is preparing data, and CPU is idle when the GPU/TPU is training the model
#  Prefetch () overlaps the preprocessing of training steps with the model execution process.
#  When the GPU is executing the Nth training step, the CPU is preparing the data of step N+1.
#  Result: shorten training time
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Rescaling and Resizing
resize_and_rescale = tf.keras.Sequential([
    # Sequential(): Single input and single output, one way to the bottom, there is only adjacent relationship between layers, no cross layer connection.
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    # edit the image into 256*256 pixels
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
    # preprocessing the data value to between 0~1
])

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #  edit the dataset images to flip, which could relief overfitting
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),  # 0.2 radian
])

# Neural Network Architecture or Model
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model =tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet", pooling='max')


model.build(input_shape=input_shape) # Builds the model based on shapes received (32,256,256,3).

print(model.summary())  # Prints a string summary (information) of the network.

# Model Compiling Using Optimizers
model.compile(
    optimizer='adam',
    # Adam combines adaptive and Momentum, the ada section refers to RMSProp.
    # The learning rate of each parameter is adjusted by an adaptive method to achieve rapid convergence.
    # In the case of large learning rate, Adam can achieve better results than SGD, but Adam is prone to fall into local optimization.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    # from_logits=False indicates the input y_pred already conforms to a certain distribution, and the system will only help you normalize the probability.
    metrics=['accuracy']
    # The accuracy of prediction

)

# Model Training
history = model.fit(
    # fit() method is to training the model, according to the set epochs.
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    # set the extent of showing the process, "1" is just showing the progress bar.
    validation_data=val_ds
    # set the validation data which will examine the model at every end of each epoch, evaluate the loss
)

scores = model.evaluate(test_ds)
# Returns the loss value & metrics values for the model in test mode. Computation is done in batches
print(scores)
print(history)
print(history.params)
print(history.history.keys)
#  history.history attribute is the record of continuous epoch training loss and evaluation value,
#  as well as verification set loss and evaluation value
print(len(history.history['accuracy']))
print(history.history['accuracy'])
acc = history.history['accuracy']
#  store the accuracy of the training data in the variable
val_acc = history.history['val_accuracy']
#  store the accuracy of the validation data in the variable
loss = history.history['loss']
#  store the loss of the training data in the variable
val_loss = history.history['val_loss']
#  store the loss of the validation data in the variable

plt.figure(figsize=(8, 8))
# create a figure which is 8 inch height, 8 inch width.

plt.subplot(1, 2, 1)
# Add an Axes to the current figure
# the value means first divide the range of origin plot into 1 row, 2 columns, and subpolt occypies its first column,
# which locates on the left-half of the image
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
# print the plot of accuracy, x-axis is the epochs, y-axis is the accuracy of the training data.
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
# print the plot of accuracy, x-axis is the epochs, y-axis is the accuracy of the validation data.
plt.legend(loc='lower right')
# set the label specification on the lower right side of the image
plt.title('Training and Validation Accuracy')
# set the plot title
plt.show()
# print the plot on the screen

# print the loss, which is similar to procedures above
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

plt.plot(range(EPOCHS), loss, label='Training Loss')

plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')
plt.show()


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    # transform the images to numpy array
    img_array = tf.expand_dims(img_array, 0)
    # increasing dimension, then it an participate in tensor model calculation
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    # find the max value of the possibility results, change it to percent,
    # then round the possibility(confidence) to a precision of 2 decimal digits.
    return predicted_class, confidence
    # return the predicted result class name and its possibility


# Showing Images with Actual Classes and Predicted Classes
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    # take one image with label to show
    for i in range(2):
        # ax = plt.subplot(3,3,i+1)
        # show two images
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        # get data and print the result in image
        actual_class = class_names[labels[i]]
        plt.title(f"Actual:{actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
        plt.show()
'''
# Saving Model
import os
model_version = max([int(i) for i in os.listdir("D:/Grade4/FYP/Working/training/models")]) + 1
#model_version = max([int(i) for i in os.listdir("../models")]) + 1
model.save(f"D:/Grade4/FYP/Working/training/models/{model_version}")
#model.save(f"../models/{model_version}")
print("Model Saving Complete")
'''