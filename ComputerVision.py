import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# importing and loading the dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# displaying a sample image
# np.set_printoptions(linewidth=200)
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])
# plt.show()

# normalizing (0 - 255 --> 0 - 1)
training_images = training_images/255.0
test_images = test_images/255.0

# defining the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(
                                        128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compiling and building the mmodel
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)


# evaluating the model
model.evaluate(test_images, test_labels)
