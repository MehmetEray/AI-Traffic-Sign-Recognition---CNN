import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    #show some images from train data
    showImages(x_train)

    #close window of figure for continue

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    plot1(model.fit(x_train, y_train, epochs=EPOCHS))
    plot2(model.fit(x_train, y_train, epochs=EPOCHS))
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    # load labels and images into arrays to return
    image_array, label_array = list(), list()

    # list all directories from specified directory argument in this function
    directories = os.listdir(data_dir)

    # load all images from directories
    for directory in directories:
        if directory == ".DS_Store":
            continue
        image_list = os.listdir(os.path.join(data_dir, directory))
        for image_name in image_list:
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(os.path.join(data_dir, image_path))

            # resize image to the specified shape
            resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # add image to the image_array
            image_array.append(resized)

            # add label of the image to the label array (directory name)
            label_array.append(int(directory))

    # return images and labels as numpy array (np.ndarray type)
    return np.array(image_array), np.array(label_array)


def get_model():
    """The model has 6 "2D Convolutional" and 1 "2D MaxPooling" layer,
    besides these; Dropout added after each Convolution and Pooling layer
    to improve learning quality for each batch, after the all process,
    BatchNormalization layer has been added before the Dense layers.
    Model type is the classic Tensorflow Sequential model.

    - Model Summary:
        After the convolutions, model's input (which is an image) shape
        converted from (30, 30, 3) to the shape of (14, 14, 16). At this
        point we apply "Flatten" function to convert feature maps into the
        shape of 1D array (shape of = (1, 3136)). After the flatten layer;
        adding 4 layers of "Dense" at the end of the model for making
        the final classification process."""

    # main Tensorflow Sequential model object
    model = tf.keras.models.Sequential([
        # define an input layer at the top of the model
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # convolutions and pooling layers
        tf.keras.layers.Conv2D(64, (5, 5), (1, 1), padding="same"),
        tf.keras.layers.Conv2D(64, (5, 5), (1, 1), padding="same"),
        tf.keras.layers.Dropout(rate=0.1),

        # second convolution and pooling layers
        tf.keras.layers.Conv2D(32, (5, 5), (1, 1), padding="valid"),
        tf.keras.layers.Conv2D(32, (5, 5), (1, 1), padding="valid"),
        tf.keras.layers.Dropout(rate=0.2),

        # final convolution and dropout layers
        tf.keras.layers.Conv2D(16, (5, 5), (1, 1), padding="valid"),
        tf.keras.layers.Conv2D(16, (5, 5), (1, 1), padding="valid"),
        tf.keras.layers.MaxPooling2D((1, 1), padding="valid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),

        # flatten the images for passing throught to dense layers
        tf.keras.layers.Flatten(),

        # dense layers for final classification
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation=tf.keras.activations.softmax)
    ])

    # compile the model - Optimizer = Adam, Loss = mean_squarred_error
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    # return final compiled model
    return model


def showImages(train_images):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # which is why you need the extra index
    plt.show()

def plot1(history):
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def plot2(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    main()
