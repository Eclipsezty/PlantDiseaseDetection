import h5py
import os
import sys
import glob
import itertools
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import confusion_matrix


from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

IMAGE_SIZE = 256
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 10
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape2",  # load dataset from filename
    shuffle=True,  # disorder the sequence of the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # 256*256 pixels image
    batch_size=BAT_SIZE  # each batch will be used to training the model at one time
)

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
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)  # the remain elements  # or use .take(int(test_split * ds_size))

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partition_tf(dataset)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)





def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes

  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                #loss='categorical_crossentropy',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

def plot_training(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()


# def train():
"""Use transfer learning and fine-tuning to train a network on a new dataset"""
nb_train_samples = train_ds
nb_classes = 4
nb_val_samples = val_ds
nb_epoch = NB_EPOCHS
batch_size = BAT_SIZE

# def train(args):
# """Use transfer learning and fine-tuning to train a network on a new dataset"""
# nb_train_samples = get_nb_files(args.train_dir)
# nb_classes = len(glob.glob(args.train_dir + "/*"))
# nb_val_samples = get_nb_files(args.val_dir)
# nb_epoch = int(args.nb_epoch)
# batch_size = int(args.batch_size)

# data prep
# train_datagen = ImageDataGenerator(
#   preprocessing_function=preprocess_input,
#   rotation_range=30,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=True
# )
# test_datagen = ImageDataGenerator(
#   preprocessing_function=preprocess_input,
#   rotation_range=30,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=True
# )

# train_generator = train_datagen.flow_from_directory(
#   args.train_dir,
#   target_size=(IM_WIDTH, IM_HEIGHT),
#   batch_size=batch_size,
# )

# validation_generator = test_datagen.flow_from_directory(
#   args.val_dir,
#   target_size=(IM_WIDTH, IM_HEIGHT),
#   batch_size=batch_size,
# )

# setup model
base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
model = add_new_last_layer(base_model, nb_classes)

# transfer learning
setup_to_transfer_learn(model, base_model)

# history_tl = model.fit(
#   train_ds,
#   epochs=nb_epoch,
#   samples_per_epoch=nb_train_samples,
#   validation_data=val_ds,
#   nb_val_samples=nb_val_samples,
#   class_weight='auto')

# fine-tuning
setup_to_finetune(model)

# history_ft = model.fit_generator(
#   train_generator,
#   samples_per_epoch=nb_train_samples,
#   nb_epoch=nb_epoch,
#   validation_data=validation_generator,
#   nb_val_samples=nb_val_samples,
#   class_weight='auto')

history_ft = model.fit(
  # fit() method is to training the model, according to the set epochs.
  train_ds,
  epochs=nb_epoch,
  batch_size=BAT_SIZE,
  verbose=1,
  # set the extent of showing the process, "1" is just showing the progress bar.
  validation_data=val_ds
  # set the validation data which will examine the model at every end of each epoch, evaluate the loss
)

model.save(f"../models/inceptionv3-ft.model")


plot_training(history_ft)



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

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, \
        classification_report
    print('Accuracy of predicting: {:.4}%'.format(accuracy_score(y_true, y_pred) * 100))
    print('Precision of predicting:{:.4}%'.format(precision_score(y_true, y_pred, average="macro") * 100))
    print('Recall of predicting:   {:.4}%'.format(
        recall_score(y_true, y_pred, average="macro") * 100))
    # print("训练数据的F1值为：", f1score_train)
    print('F1 score:', f1_score(y_true, y_pred, average="macro"))
    print('Cohen\'s Kappa coefficient: ', cohen_kappa_score(y_true, y_pred))
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
    #         recall += float(accu[i]) / column[i]
    # recall = recall / 4
    # for i in range(0, 4):
    #     if row[i] != 0:
    #         precision += float(accu[i]) / row[i]
    # precision = precision / 4
    # f1_score = (2 * (precision * recall)) / (precision + recall)
    # print("recall: ",recall, "  precision:  ",precision,"  f1_socre: " ,f1_score)


    plt.show()


def generate_confusion_matrix():
    labels = np.concatenate([y for x, y in test_ds], axis=0)
    predict_classes = model.predict(test_ds)
    print(predict_classes)
    true_classes = np.argmax(predict_classes, 1)
    print(true_classes)
    plot_confusion_matrix(labels, true_classes,save_flg=False)
    # plot_confusion_matrix(true_classes,labels, save_flg=False)

generate_confusion_matrix()


