import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# importing and loading the dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(training_images[42])
print(training_labels[42])
print(training_images[42])
