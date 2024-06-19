import inspect
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import CutMix
import cv2
from keras_preprocessing.image import load_img,img_to_array
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
IMAGE_SIZE = 256
BATCH_SIZE = 32

randomCropSize = np.random.randint(128,256)

#print("Dataset Making")
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Grape",  # load dataset from filename
    shuffle=True,  # disorder the sequence of the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # 256*256 pixels image
    batch_size=BATCH_SIZE  # each batch will be used to training the model at one time
  )

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #  edit the dataset images to flip, which could relief overfitting
    tf.keras.layers.RandomRotation(0.2),  # 0.2 radian
    tf.keras.layers.RandomCrop(randomCropSize,randomCropSize)
])





def printInfo():

  # Printing Class Names
  print("Class Names")
  class_names = dataset.class_names
  print(class_names)

  print("Length of Dataset")
  print(len(dataset))

  for image_batch, label_batch in dataset.take(1):  # First Batch
    print(image_batch.shape)  # （batch size, x_pixel, y_pixel, channel)
    print(label_batch.numpy())  # print the label of images (whether infected diseases)


def blend(image1, image2, factor=0.5):
  '''Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 'extrapolates' the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  '''
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1,tf.float32)
  #image1 = tf.to_float(image1)
  image2 = tf.cast(image2,tf.float32)
  #image2 = tf.to_float(image2)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1,tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def testblend():
  for image_batch, label_batch in dataset.take(1):  # Batch 1
    # for i in range(1):  # Print one image
    p1 = image_batch[1].numpy().astype("uint8")
  plt.subplot(1, 3, 1)
  plt.imshow(p1)
  # "uint8" is from 0-255
  plt.axis("off")  # hide the axis
  p2 = image_batch[2].numpy().astype("uint8")
  plt.subplot(1, 3, 2)
  plt.imshow(p2)
  plt.axis("off")
  p3 = blend(p1,p2,0.5)
  plt.subplot(1, 3, 3)
  plt.imshow(p3)
  plt.title("blended")
  plt.axis("off")
  plt.show()


def testCutMix():
  for image_batch, label_batch in dataset.take(1):
    input_image = image_batch[0]
  # Convert image_batch to numpy array
  image_batch = np.array(image_batch)
  # Conver image_batch_labels to numpy array
  image_batch_labels = np.array(label_batch)

  # Show original images
  print("Original Images")
  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      img=image_batch[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("Original Image")
      plt.axis("off")
  plt.show()

  image_batch_updated, image_batch_labels_updated = CutMix.cutMix(image_batch, image_batch_labels,1.0)


  # Show CutMix images
  print("CutMix Images")
  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      plt.imshow((image_batch_updated[2 * i + j])/255)
      plt.title("CutMix Image")
      plt.axis("off")
  plt.show()
  # Print labels
  print('Original labels:')
  print(label_batch)
  print('Updated labels')
  print(image_batch_labels_updated)


def brightness(image, factor=0.5):
  '''Equivalent of PIL Brightness.'''
  degenerate = (tf.zeros_like(image))+np.random.randint(32,128)
  return blend(degenerate, image, factor)


def testBrightness():
  for image_batch, label_batch in dataset.take(1):
    img= image_batch[1].numpy().astype("uint8")
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    img2=brightness(img,0.5)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("brightness")
    plt.axis("off")
  plt.show()


def testShadow():
  for image_batch, label_batch in dataset.take(1):
    input_image = image_batch[0]
  # Convert image_batch to numpy array
  image_batch = np.array(image_batch)
  # Conver image_batch_labels to numpy array
  image_batch_labels = np.array(label_batch)

  # Show original images
  print("Original Images")
  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      img=image_batch[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("Original Image")
      plt.axis("off")
  plt.show()

  image_batch_updated, image_batch_labels_updated = CutMix.shadowCutMix(image_batch, image_batch_labels,1.0)

  # Show CutMix images
  print("Images with Shadow")
  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      plt.imshow((image_batch_updated[2 * i + j])/255)
      plt.title("Images with Shadow")
      plt.axis("off")
  plt.show()
  # Print labels
  # print('Original labels:')
  # print(label_batch)
  # print('Updated labels')
  # print(image_batch_labels_updated)

def TestDA():
  for image_batch, label_batch in dataset.take(1):
    input_image = image_batch[0]

  for i in range(1):
    for j in range(3):
      plt.subplot(2, 3, 2 * i + j+1)
      img=image_batch[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("before")
      plt.axis("off")
  plt.show()

  image_batch = data_augmentation(image_batch)
  for i in range(1):
    for j in range(3):
      plt.subplot(2, 3, 2 * i + j+1)
      img=image_batch[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("after")
      plt.axis("off")
  plt.show()

TestDA()

def toGrey(image):

    #image = img_to_array(image)
    image = image.astype(dtype='uint8')
    #image = image.astype(dtype='float32')
    #plt.subplot(1,2,1)
    #plt.imshow(image)
    image = tf.image.rgb_to_grayscale(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = tf.squeeze(image, 2)
    #plt.subplot(1,2,2)
    #plt.imshow(image)
    #plt.show()
    return image
    #return image,label

def tgrey():
  for image_batch, label_batch in dataset.take(1):
    input_image = image_batch[0]

  image_batch = np.array(image_batch)

  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      img=image_batch[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("before")
      plt.axis("off")
  plt.show()
  image_batch_updated = image_batch

  image_batch_updated2 = toGrey(image_batch_updated)
  for i in range(2):
    for j in range(2):
      plt.subplot(2, 2, 2 * i + j + 1)
      img=image_batch_updated2[2 * i + j]
      img=img/255
      plt.imshow(img)
      plt.title("after")
      plt.axis("off")
  plt.show()

tgrey()


def testgrey(image_batch):
  image_batch = np.array(image_batch)
  image_batch_updated = image_batch
    #for i in range(len(image_batch)):
      #image_batch_updated[i] = toGrey(image_batch_updated[i].numpy().astype("uint8"))
  image_batch_updated = toGrey(image_batch_updated)
  return image_batch_updated


def shadowCutMix1(image_batch, image_batch_labels, beta=1.0):

    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    bbx1, bby1, bbx2, bby2 = CutMix.rand_bbox(image_batch[0].shape, lam)
    image_batch_updated = image_batch
    # image_batch_updated2 = brightness(image_batch_updated)
    image_batch_updated2 = image_batch_updated
    image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = image_batch_updated2[:, bbx1:bbx2, bby1:bby2, :]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))

    return image_batch_updated















