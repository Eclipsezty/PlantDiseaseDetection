import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.models import Model

sdir = r'F:/Study/FYP/training/Grape3'
working_dir = r'./'  # directory to store augmented images

min_samples = 40  # set limit for minimum images a class must have to be included in the dataframe
max_samples = 1000  # since each class has more than 200 images all classes will be trimmed to have 200 images per class
num_samples = 1000  # number of samples in each class exactly
epochs = 40
image_size = (200, 200)  # size of augmented images
batch_size = 20

file_paths = []
labels = []
class_list = os.listdir(sdir)
for CLASS in class_list:
    class_path = os.path.join(sdir, CLASS)
    file_list = os.listdir(class_path)
    if len(file_list) >= min_samples:
        for file in file_list:
            file_path = os.path.join(class_path, file)
            file_paths.append(file_path)
            labels.append(CLASS)
    else:
        print('class ', CLASS, ' has only', len(file_list), ' samples and will not be included in dataframe')
Files = pd.Series(file_paths, name='filepaths')
Labels = pd.Series(labels, name='labels')
dataframe = pd.concat([Files, Labels], axis=1)
train_dataframe, val_test_df = train_test_split(dataframe, train_size=.9, shuffle=True, random_state=123, stratify=dataframe['labels'])
val_dataframe, test_dataframe = train_test_split(val_test_df, train_size=.5, shuffle=True, random_state=123,
                                                 stratify=val_test_df['labels'])

Class = sorted(list(train_dataframe['labels'].unique()))
c_num = len(Class)
groups = train_dataframe.groupby('labels')

def trim_dataset(dataframe, max_num, min_num, COLUMN):
    dataframe = dataframe.copy()
    G = dataframe.groupby(COLUMN)
    trim_dataframe = pd.DataFrame(columns=dataframe.columns)
    G = dataframe.groupby(COLUMN)
    for label in dataframe[COLUMN].unique():
        Group = G.get_group(label)
        group_num = len(Group)
        if group_num > max_num:
            group_sample = Group.sample(n=max_num, random_state=123, axis=0)
            trim_dataframe = pd.concat([trim_dataframe, group_sample], axis=0)
        else:
            if group_num >= min_num:
                group_sample = Group
                trim_dataframe = pd.concat([trim_dataframe, group_sample], axis=0)
    return trim_dataframe


column = 'labels'
train_dataframe = trim_dataset(train_dataframe, max_samples, min_samples, column)


def balance_dataset(dataframe, target_num, dir, img_size):
    dataframe = dataframe.copy()
    gen_dir = os.path.join(dir, 'aug')
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.mkdir(gen_dir)
    for label in dataframe['labels'].unique():
        path = os.path.join(gen_dir, label)
        os.mkdir(path)
    imgCount = 0
    imgGen = ImageDataGenerator(horizontal_flip=True, rotation_range=30, width_shift_range=.2,
                                height_shift_range=.1, zoom_range=.1)
    groups = dataframe.groupby('labels')
    for label in dataframe['labels'].unique():
        G = groups.get_group(label)
        sample_num = len(G)
        if sample_num < target_num:
            aug_image_num = 0
            gen_count = target_num - sample_num
            t_dir = os.path.join(gen_dir, label)
            augmented_gen = imgGen.flow_from_dataframe(G, x_col='filepaths', y_col=None, target_size=img_size,
                                                 class_mode=None, batch_size=1, shuffle=False,
                                                 save_to_dir=t_dir, save_prefix='aug-', color_mode='rgb',
                                                 save_format='jpg')
            while aug_image_num < gen_count:
                imgs = next(augmented_gen)
                aug_image_num += len(imgs)
            imgCount += aug_image_num

    aug_filepaths = []
    aug_labels = []
    classeslist = os.listdir(gen_dir)
    for CLASS in classeslist:
        class_path = os.path.join(gen_dir, CLASS)
        file_list = os.listdir(class_path)
        for file in file_list:
            file_path = os.path.join(class_path, file)
            aug_filepaths.append(file_path)
            aug_labels.append(CLASS)
    Filepaths = pd.Series(aug_filepaths, name='filepaths')
    Labels = pd.Series(aug_labels, name='labels')
    aug_dataframe = pd.concat([Filepaths, Labels], axis=1)
    dataframe = pd.concat([dataframe, aug_dataframe], axis=0).reset_index(drop=True)
    return dataframe


train_dataframe = balance_dataset(train_dataframe, num_samples, working_dir, image_size)

tr_gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                            height_shift_range=.2, zoom_range=.2)
test_and_val_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_dataframe, x_col='filepaths', y_col='labels', target_size=image_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

val_gen = test_and_val_gen.flow_from_dataframe(val_dataframe, x_col='filepaths', y_col='labels', target_size=image_size,
                                               class_mode='categorical', color_mode='rgb', shuffle=False,
                                               batch_size=batch_size)

Length = len(test_dataframe)
test_batchsize = \
sorted([int(Length / n) for n in range(1, Length + 1) if Length % n == 0 and Length / n <= 80], reverse=True)[0]

test_gen = test_and_val_gen.flow_from_dataframe(test_dataframe, x_col='filepaths', y_col='labels', target_size=image_size,
                                                class_mode='categorical', color_mode='rgb', shuffle=False,
                                                batch_size=test_batchsize)
Class = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
c_num = len(Class)
labels = test_gen.labels


image_shape = (image_size[0], image_size[1], 3)
model_base = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",
                                                               input_shape=image_shape, pooling='max')
model_base.trainable = True
o = model_base.output
o = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(o)
o = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(o)
o = Dropout(rate=.4, seed=123)(o)
output = Dense(c_num, activation='softmax')(o)
model = Model(inputs=model_base.input, outputs=output)
lr = .001
model.summary()
model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=val_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)


def training_plot(training_datas, start_epoch):

    trainAcc = training_datas.history['accuracy']
    trainLoss = training_datas.history['loss']
    valAcc = training_datas.history['val_accuracy']
    valLoss = training_datas.history['val_loss']
    Epoch_num = len(trainAcc) + start_epoch
    EPOCH = []
    for i in range(start_epoch, Epoch_num):
        EPOCH.append(i + 1)
    i_loss = np.argmin(valLoss)
    val_lowestloss = valLoss[i_loss]
    i_acc = np.argmax(valAcc)
    acc_h = valAcc[i_acc]
    plt.style.use('fivethirtyeight')
    s_label = 'best epoch= ' + str(i_loss + 1 + start_epoch)
    v_label = 'best epoch= ' + str(i_acc + 1 + start_epoch)
    figure, Ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    Ax[0].plot(EPOCH, trainLoss, 'r', label='Training loss')
    Ax[0].plot(EPOCH, valLoss, 'g', label='Validation loss')
    Ax[0].scatter(i_loss + 1 + start_epoch, val_lowestloss, s=150, c='blue', label=s_label)
    Ax[0].set_title('Training and Validation Loss')
    Ax[0].set_xlabel('Epochs')
    Ax[0].set_ylabel('Loss')
    Ax[0].legend()
    Ax[1].plot(EPOCH, trainAcc, 'r', label='Training Accuracy')
    Ax[1].plot(EPOCH, valAcc, 'g', label='Validation Accuracy')
    Ax[1].scatter(i_acc + 1 + start_epoch, acc_h, s=150, c='blue', label=v_label)
    Ax[1].set_title('Training and Validation Accuracy')
    Ax[1].set_xlabel('Epochs')
    Ax[1].set_ylabel('Accuracy')
    Ax[1].legend()
    plt.tight_layout()
    plt.show()


training_plot(history, 0)


def predict(t_gen):
    y_p = []
    y_t = t_gen.labels
    Class = list(t_gen.class_indices.keys())
    class_num = len(Class)
    errors = 0
    preds = model.predict(t_gen, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        predict_index = np.argmax(p)
        true_index = t_gen.labels[i]
        if predict_index != true_index:
            errors = errors + 1
        y_p.append(predict_index)

    yp = np.array(y_p)
    yt = np.array(y_t)
    if class_num <= 30:
        C = confusion_matrix(yt, yp)

        plt.figure(figsize=(12, 9))
        sns.heatmap(C, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_num) + 0.5, Class, rotation=90)
        plt.yticks(np.arange(class_num) + 0.5, Class, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
    cr = classification_report(y_t, y_p, target_names=Class, digits=4)
    print(cr)

predict(test_gen)

save_path = "F:/Study/FYP/training/models"
save_name = "Grape3-EfficientNet_e40" + '.h5'
m_save_path = os.path.join(save_path, save_name)
model.save(m_save_path)
print('model was saved in ', m_save_path)
