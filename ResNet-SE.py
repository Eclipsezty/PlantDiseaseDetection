# Packages to import
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
import itertools

from keras.applications.imagenet_utils import decode_predictions
from sklearn.metrics import confusion_matrix

is_keras_tensor = K.is_keras_tensor

from SE import squeeze_excite_block
from utils import _obtain_input_shape, get_source_inputs, _tensor_shape
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                              GlobalAveragePooling2D, GlobalMaxPooling2D,
                              Input, MaxPooling2D, add)
from keras import layers
from keras.layers import Lambda




import matplotlib.pyplot as plt

# Defining Constants

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 30


print("Dataset Making")
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # "PlantVillage",  # load dataset from filename
    "Grape2",  # load dataset from filename
    shuffle=True,  # disorder the sequence of the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # 256*256 pixels image
    batch_size=BATCH_SIZE  # each batch will be used to training the model at one time
)

class_names = dataset.class_names

# Dataset Partitioning
def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=1000):
    ds_size = len(ds)
    if shuffle:
        #ds = ds.shuffle(shuffle_size, seed=12)
        ds = ds.shuffle(shuffle_size)

    train_size = int(train_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    return train_ds, val_ds



train_ds, val_ds = get_dataset_partition_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Rescaling and Resizing
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
])



# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),  # 0.2 radian
    tf.keras.layers.RandomCrop(191,191),
])



# Based on ResNet50
def SEResNet(input_shape =None,
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=4):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, bottleneck, weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

    # load weights

    return model




def _create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, bottleneck, weight_decay, pooling):
    """Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x



def _resnet_bottleneck_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block with bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m



def _resnet_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block without bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m









# Neural Network Architecture or Model
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 4
#
# model = models.Sequential([
#     resize_and_rescale,
#     data_augmentation,
#     SEResNet()
# ])
model = SEResNet()

model.build(input_shape=input_shape)

print(model.summary())


model.compile(
    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']

)

# Model Training
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)

print(history)
print(history.params)
print(history.history.keys)
print(len(history.history['accuracy']))
print(history.history['accuracy'])

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

plot_training(history)


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
    plt.show()

def generate_confusion_matrix():
    labels = np.concatenate([y for x, y in val_ds], axis=0)
    predict_classes = model.predict(val_ds)
    print(predict_classes)
    true_classes = np.argmax(predict_classes, 1)
    print(true_classes)
    # plot_confusion_matrix(true_classes,labels , save_flg=False)
    plot_confusion_matrix(labels ,true_classes, save_flg=False)

generate_confusion_matrix()
'''
# Saving Model
import os
model_version = max([int(i) for i in os.listdir("D:/Grade4/FYP/Working/training/models")]) + 1
#model_version = max([int(i) for i in os.listdir("../models")]) + 1
model.save(f"D:/Grade4/FYP/Working/training/models/{model_version}")
#model.save(f"../models/{model_version}")
print("Model Saving Complete")
'''