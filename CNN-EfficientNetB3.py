import numpy as np
import pandas as pd
import os
import matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
matplotlib.use('TkAgg')
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

sdir = r'F:\Research\CAMIC\New CAMIC\Train'
working_dir = r'./'  # directory to store augmented images

min_samples = 40  # set limit for minimum images a class must have to be included in the dataframe
max_samples = 1000  # since each class has more than 200 images all classes will be trimmed to have 200 images per class
num_samples = 1000  # number of samples in each class exactly
epochs = 30
img_size = (200, 200)  # size of augmented images
batch_size = 20

filepaths = []
labels = []
class_list = os.listdir(sdir)
for CLASS in class_list:
    classpath = os.path.join(sdir, CLASS)
    filelist = os.listdir(classpath)
    if len(filelist) >= min_samples:
        for f in filelist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(CLASS)
    else:
        print('class ', CLASS, ' has only', len(filelist), ' samples and will not be included in dataframe')
Files = pd.Series(filepaths, name='filepaths')
Labels = pd.Series(labels, name='labels')
df = pd.concat([Files, Labels], axis=1)
train_df, val_test_df = train_test_split(df, train_size=.9, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df = train_test_split(val_test_df, train_size=.5, shuffle=True, random_state=123,
                                     stratify=val_test_df['labels'])

print('train_df lenght: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
# get the number of classes and the images count for each class in train_df
Class = sorted(list(train_df['labels'].unique()))
Class_num = len(Class)
print('The number of classes is: ', Class_num)
groups = train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
count_list = []
class_list = []
for label in sorted(list(train_df['labels'].unique())):
    group = groups.get_group(label)
    count_list.append(len(group))
    class_list.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

# get the classes with the minimum and maximum number of train images
max_imgNum = np.max(count_list)
max_index = count_list.index(max_imgNum)
max_class = class_list[max_index]
min_value = np.min(count_list)
min_index = count_list.index(min_value)
min_class = class_list[min_index]
print(max_class, ' has the most images= ', max_imgNum, ' ', min_class, ' has the least images= ', min_value)


def trim_dataset(dataframe, max_samples, min_samples, column):
    dataframe = dataframe.copy()
    groups = dataframe.groupby(column)
    trimmed_df = pd.DataFrame(columns=dataframe.columns)
    groups = dataframe.groupby(column)
    for label in dataframe[column].unique():
        group = groups.get_group(label)
        count = len(group)
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if count >= min_samples:
                sampled_group = group
                trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
    print('after trimming, the maximum samples in any class is now ', max_samples,
          ' and the minimum samples in any class is ', min_samples)
    return trimmed_df


column = 'labels'
train_df = trim_dataset(train_df, max_samples, min_samples, column)


def balance_dataset(dataframe, n, working_dir, img_size):
    dataframe = dataframe.copy()
    print('Initial length of dataframe is ', len(dataframe))
    aug_dir = os.path.join(working_dir, 'aug')  # directory to store augmented images
    if os.path.isdir(aug_dir):  # start with an empty directory
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in dataframe['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)  # make class directories within aug directory
    # create and store the augmented images
    imgCount = 0
    imgGen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                                height_shift_range=.2, zoom_range=.2)
    groups = dataframe.groupby('labels')  # group by class
    for label in dataframe['labels'].unique():  # for every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count = len(group)  # determine how many samples there are in this class
        if sample_count < n:  # if the class has less than target number of images
            aug_img_count = 0
            delta = n - sample_count  # number of augmented images to create
            target_dir = os.path.join(aug_dir, label)  # define where to write the images
            msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            print(msg, '\r', end='')  # prints over on the same line
            aug_gen = imgGen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
                                                 class_mode=None, batch_size=1, shuffle=False,
                                                 save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                                 save_format='jpg')
            while aug_img_count < delta:
                imgs = next(aug_gen)
                aug_img_count += len(imgs)
            imgCount += aug_img_count
    print('Total Augmented images created= ', imgCount)
    # create aug_df and merge with train_df to create composite training set ndf
    aug_filepaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for CLASS in classlist:
        classpath = os.path.join(aug_dir, CLASS)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_filepaths.append(fpath)
            aug_labels.append(CLASS)
    Filepaths = pd.Series(aug_filepaths, name='filepaths')
    Labels = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Filepaths, Labels], axis=1)
    dataframe = pd.concat([dataframe, aug_df], axis=0).reset_index(drop=True)
    print('Length of augmented dataframe is now ', len(dataframe))
    return dataframe


train_df = balance_dataset(train_df, num_samples, working_dir, img_size)

trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                           height_shift_range=.2, zoom_range=.2)
t_and_v_gen = ImageDataGenerator()
msg = '{0:70s} for train generator'.format(' ')
print(msg, '\r', end='')  # prints over on the same line
train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
msg = '{0:70s} for valid generator'.format(' ')
print(msg, '\r', end='')  # prints over on the same line
valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                            class_mode='categorical', color_mode='rgb', shuffle=False,
                                            batch_size=batch_size)
# this insures that going through all the sample in the test set exactly once.
length = len(test_df)
test_batch_size = \
sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
msg = '{0:70s} for test generator'.format(' ')
print(msg, '\r', end='')  # prints over on the same line
test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=False,
                                           batch_size=test_batch_size)
# from the generator it can get information which will be needed later
Class = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
Class_num = len(Class)
labels = test_gen.labels
print('test batch size: ', test_batch_size, ' number of classes : ', Class_num)


def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)  # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 25:  # show maximum of 25 images
        r = length
    else:
        r = 25
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i] / 255
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=14)
        plt.axis('off')
    plt.show()


# show_image_samples(train_gen)

img_shape = (img_size[0], img_size[1], 3)
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",
                                                               input_shape=img_shape, pooling='max')
base_model.trainable = True
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.4, seed=123)(x)
output = Dense(Class_num, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
lr = .001  # start with this learning rate
# model.summary()
model.compile(Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)


def training_plot(training_data, start_epoch):
    # Plot the training and validation data
    trainAcc = training_data.history['accuracy']
    trainLoss = training_data.history['loss']
    valAcc = training_data.history['val_accuracy']
    valLoss = training_data.history['val_loss']
    Epoch_num = len(trainAcc) + start_epoch
    EPOCH = []
    for i in range(start_epoch, Epoch_num):
        EPOCH.append(i + 1)
    index_loss = np.argmin(valLoss)  # this is the epoch with the lowest validation loss
    val_lowestloss = valLoss[index_loss]
    index_acc = np.argmax(valAcc)
    acc_highest = valAcc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(EPOCH, trainLoss, 'r', label='Training loss')
    axes[0].plot(EPOCH, valLoss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowestloss, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(EPOCH, trainAcc, 'r', label='Training Accuracy')
    axes[1].plot(EPOCH, valAcc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


training_plot(history, 0)


def predict(test_gen):
    y_pred = []
    y_true = test_gen.labels
    Class = list(test_gen.class_indices.keys())
    class_num = len(Class)
    errors = 0
    preds = model.predict(test_gen, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        predict_index = np.argmax(p)
        true_index = test_gen.labels[i]  # labels are integer values
        if predict_index != true_index:  # a misclassification has occurred
            errors = errors + 1
        y_pred.append(predict_index)

    acc = (1 - errors / tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    if class_num <= 30:
        cm = confusion_matrix(ytrue, ypred)
        # plot the confusion matrix
        plt.figure(figsize=(12, 9))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_num) + 0.5, Class, rotation=90)
        plt.yticks(np.arange(class_num) + 0.5, Class, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=Class, digits=4)  # create classification report
    print("Classification Report:\n----------------------\n", clr)

predict(test_gen)

save_path = "F:/Study/FYP/training/models"
save_name = "Rice-EfficientNet_e30" + '.h5'
model_save_path = os.path.join(save_path, save_name)
model.save(model_save_path)
print('model was saved as ', model_save_path)
