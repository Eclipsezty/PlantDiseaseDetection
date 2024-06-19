# Packages to import
import itertools
import os
import numpy as np
import sklearn as sk
import tensorflow as tf

import ActivationFunction
import DataAugmentation
from keras import models
from keras import layers
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from keras_preprocessing.image import load_img, img_to_array
from keras import backend as K
from sklearn.metrics import confusion_matrix


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



# Defining Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3  # RGB three kinds of color
EPOCHS = 5  # one epoch means the model ergodic the whole dataset
datasetSize = 4062

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape3",  # load dataset from filename
    shuffle=True,  # disorder the sequence of the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # 256*256 pixels image
    batch_size=BATCH_SIZE  # each batch will be used to training the model at one time
)

# labels = np.concatenate([y for x, y in dataset], axis=0)


def balance(df, n, working_dir, img_size):
    df=df.copy()
    print('Initial length of dataframe is ', len(df))
    aug_dir=os.path.join(working_dir, 'aug')# directory to store augmented images
    if os.path.isdir(aug_dir):# start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df['labels'].unique():
        dir_path=os.path.join(aug_dir,label)
        os.mkdir(dir_path) # make class directories within aug directory
    # create and store the augmented images
    total=0
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                  height_shift_range=.2, zoom_range=.2)
    groups=df.groupby('labels') # group by class
    for label in df['labels'].unique():  # for every class
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        if sample_count< n: # if the class has less than target number of images
            aug_img_count=0
            delta=n - sample_count  # number of augmented images to create
            target_dir=os.path.join(aug_dir, label)  # define where to write the images
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
    print('Total Augmented images created= ', total)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_fpaths=[]
    aug_labels=[]
    classlist=os.listdir(aug_dir)
    for klass in classlist:
        classpath=os.path.join(aug_dir, klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            aug_fpaths.append(fpath)
            aug_labels.append(klass)
    Fseries=pd.Series(aug_fpaths, name='filepaths')
    Lseries=pd.Series(aug_labels, name='labels')
    aug_df=pd.concat([Fseries, Lseries], axis=1)
    df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(df))
    return df

class_names = dataset.class_names

#*********Grey Scale*************
#dataset = dataset.map(lambda x,y: DataAugmentation.toGrey(x,y))
# for image_batch in dataset:
#     image = DataAugmentation.testgrey(image_batch)


def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    #  the 80% of dataset is used in training, 10% is used to validate whether the model is valid or overfitting
    #  using a buffer to shuffle, which size is 10000. Firstly set the first 10000 elements in the buffer, and pop, push randomly
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        # use random seed to store the random number, allow running code multiple times can get the same result

    train_size = int(train_split * ds_size)  # size of dataset * 0.8
    val_size = int(val_split * ds_size)  # size of dataset * 0.1

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds


train_ds, val_ds= get_dataset_partition_tf(dataset)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Neural Network Architecture or Model
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 4


model = models.Sequential([

    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    layers.Conv2D(32, (3, 3), activation= "relu" ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(n_classes, activation='softmax')
])

# model = models.Sequential([
#     resize_and_rescale,
#     data_augmentation,
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(n_classes, activation='softmax')
# ])


#model.build(input_shape=[input_shape,input_shape])
model.build(input_shape = input_shape)

print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
    # cm = sk.metrics.confusion_matrix()

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

scores = model.evaluate(val_ds)
# Returns the loss value & metrics values for the model in test mode. Computation is done in batches
print(scores)
print(history)
print(history.params)
print(history.history.keys)
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


#
# def predict(model, img):
#     img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#     img_array = tf.expand_dims(img_array, 0)
#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)
#     return predicted_class, confidence
# for images, labels in val_ds.take(1):
#     for i in range(2):
#         plt.imshow(images[i].numpy().astype("uint8"))
#
#         predicted_class, confidence = predict(model, images[i].numpy())
#         actual_class = class_names[labels[i]]
#         plt.title(f"Actual:{actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
#         plt.axis("off")
#         #plt.show()


#
# def plot_training(history):
#   acc = history.history['accuracy']
#   val_acc = history.history['val_accuracy']
#   loss = history.history['loss']
#   val_loss = history.history['val_loss']
#   epochs = range(len(acc))
#
#   plt.plot(epochs, acc, 'r.')
#   plt.plot(epochs, val_acc, 'r')
#   plt.title('Training and validation accuracy')
#
#   plt.figure()
#   plt.plot(epochs, loss, 'r.')
#   plt.plot(epochs, val_loss, 'r-')
#   plt.title('Training and validation loss')
#   plt.show()
#
# plot_training(history)

def tr_plot(tr_data, start_epoch):
    # Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    plt.show()


tr_plot(history, 0)


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix",
                          cmap=plt.cm.Blues, save_flg=False):
    classes = [str(i) for i in range(4)]  # 参数i的取值范围根据你自己数据集的划分类别来修改，我这儿为7代表数据集共有7类
    labels = range(4)  # 数据集的标签类别，跟上面I对应
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    # print(cm[3,3])
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    if save_flg:
        plt.savefig("./confusion_matrix.png")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,classification_report
    print('Accuracy of predicting: {:.4}%'.format(accuracy_score(y_true, y_pred) * 100))
    print('Precision of predicting:{:.4}%'.format(precision_score(y_true, y_pred,average="macro") * 100))
    print('Recall of predicting:   {:.4}%'.format(
        recall_score(y_true, y_pred,average="macro") * 100))
    # print("训练数据的F1值为：", f1score_train)
    print('F1 score:',f1_score(y_true, y_pred,average="macro"))
    print('Cohen\'s Kappa coefficient: ',cohen_kappa_score(y_true, y_pred))
    print('Classification report:\n', classification_report(y_true, y_pred))



    #
    # accu = [0, 0, 0, 0]
    # column = [0, 0, 0, 0]
    # row = [0, 0, 0, 0]
    # dataNum = 0
    # accuracy = 0
    # recall = 0
    # precision = 0
    # for i in range(0, 4):
    #     accu[i] = cm[i][i]
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         column[i] += cm[j][i]
    # for i in range(0, 4):
    #     dataNum += column[i]
    #     for j in range(0, 4):
    #         row[i] += cm[i][j]
    # for i in range(0, 4):
    #     accuracy += float(accu[i]) / dataNum
    # for i in range(0, 4):
    #     if column[i] != 0:
    #         precision += float(accu[i]) / column[i]
    # precision = precision / 4
    # for i in range(0, 4):
    #     if row[i] != 0:
    #         recall += float(accu[i]) / row[i]
    # recall = recall / 4
    # f1_score = (2 * (recall * precision)) / (recall + precision)
    # print("recall: ",recall, "  precision:  ",precision,"  f1_socre: " ,f1_score)

    plt.show()


def generate_confusion_matrix():
    labels = np.concatenate([y for x, y in val_ds], axis=0)
    predict_classes = model.predict(val_ds)
    print(predict_classes)
    true_classes = np.argmax(predict_classes, 1)
    print(true_classes)
    # plot_confusion_matrix(true_classes,labels , save_flg=False)
    plot_confusion_matrix(labels,true_classes,  save_flg=False)

generate_confusion_matrix()

save_path = "F:/Study/FYP/training/models"
save_id= "Grape3-Eff_e10s"+ '.h5'
model_save_loc=os.path.join(save_path, save_id)
#model_save_loc=os.path.join(working_dir, save_id)
model.save(model_save_loc)
print ('model was saved as ' , model_save_loc )




# def predict2(model, img):
#     img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#     img_array = tf.expand_dims(img_array, 0)
#     predictions = model.predict(img_array)
#     return predictions
# for images, labels in test_ds.take(1):
#     for i in range(2):
#         plt.imshow(images[i].numpy().astype("uint8"))
#
#         predictions = predict2(model, images[i].numpy())
#         print("**************************")
#         print(predictions)
#         actual_class = class_names[labels]
#         print(labels)
