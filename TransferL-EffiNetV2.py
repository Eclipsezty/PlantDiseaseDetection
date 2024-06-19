import random
import numpy as np
import tensorflow as tf
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

import os

from keras.applications import EfficientNetV2S
from keras import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

img_size=(200,200)
num_classes = 4
img_shape=(img_size[0], img_size[1], 3)
epochs=5





model = Sequential()
model.add(EfficientNetV2S(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max'))
model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


data_generator = ImageDataGenerator()


train_generator = data_generator.flow_from_directory(
        'F:/Study/FYP/training/Grape3',
        target_size=(img_size[0], img_size[1]),
        batch_size=32,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'F:/Study/FYP/training/Grape3-val',
        target_size=(img_size[0], img_size[1]),
        batch_size=20,
        class_mode='categorical')

# model.fit(
#         train_generator,
#         steps_per_epoch=6,
#         validation_data=validation_generator,
#         validation_steps=1)

history=model.fit(x=train_generator,  epochs=epochs, verbose=1,  validation_data=validation_generator,
               validation_steps=1)

#*********19
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

#*******20
def predictor(test_gen, test_steps):
    y_pred = []
    y_true = test_gen.labels
    classes = list(test_gen.class_indices.keys())
    class_count = len(classes)
    errors = 0
    preds = model.predict(test_gen, verbose=1)
    tests = len(preds)
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = test_gen.labels[i]  # labels are integer values
        if pred_index != true_index:  # a misclassification has occurred
            errors = errors + 1
        y_pred.append(pred_index)

    acc = (1 - errors / tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    if class_count <= 30:
        cm = confusion_matrix(ytrue, ypred)
        # plot the confusion matrix
        plt.figure(figsize=(12, 9))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count) + 1.5, classes, rotation=90)
        plt.yticks(np.arange(class_count) + 1.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits=4)  # create classification report
    print("Classification Report:\n----------------------\n", clr)
    return errors, tests


save_path = "F:/Study/FYP/training/models"
save_id= "EffNetV2_Grape3_e" + epochs+ '.h5'
model_save_loc=os.path.join(save_path, save_id)
#model_save_loc=os.path.join(working_dir, save_id)
model.save(model_save_loc)
print ('model was saved as ' , model_save_loc )