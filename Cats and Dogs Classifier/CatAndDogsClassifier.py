import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file(
    'cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# print(os.path.dirname(path_to_zip))

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
# directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

# print('total training cat images:', num_cats_tr)
# print('total training dog images:', num_dogs_tr)

# print('total validation cat images:', num_cats_val)
# print('total validation dog images:', num_dogs_val)
# print("--")
# print("Total training images:", total_train)
# print("Total validation images:", total_val)

# variables to use while pre-processing the dataset and training the network
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Generator for our training data
train_image_generator = ImageDataGenerator(
    rescale=1./255)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(
    rescale=1./255)

# flow_from_directory method load images from the disk, applies rescaling, and resizes the images into the required dimensions
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

# next function returns a batch from the dataset. The return value of next function is in form of (x_train, y_train)
sample_training_images, _ = next(train_data_gen)


# function to plot images in the form of a grid with 1 row and 5 columns where images are placed in each column


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plotImages(sample_training_images[:5])

# creating the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(
        IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True), metrics=['accuracy'])

model.summary()

# training the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
