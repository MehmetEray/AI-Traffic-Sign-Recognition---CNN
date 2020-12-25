import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow import keras
import cv2

data = []
labels = []
cur_path = "/Users/mehmeteraysurmeli/Downloads/"
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
cwd = os.getcwd()

def main():
    global labels, data

    # Get image arrays and labels for all image files
    load_data(cur_path)
    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    # Splitting training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state=42)
    # showImages(x_train)
    get_model(x_train, y_train, x_test, y_test)


def load_data(data_dir):
    # Retrieving the images and their labels
    for i in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, 'gtsrb', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(path + '/' + a)
                image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image = np.array(image)
                # sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")


def showImages(train_images):
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
    plt.show()


def get_model(X_train, y_train, X_test, y_test):
    y_train = to_categorical(y_train, NUM_CATEGORIES)
    y_test = to_categorical(y_test, NUM_CATEGORIES)

    # Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.summary()

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    model.summary()
    #
    opt = keras.optimizers.Adam(learning_rate=0.001)  # overfit olmamasi icin
    #
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_test, y_test))
    model.save("my_model_cnn.h5")

    # plot1(history)
    print("=========================================================")
    # plot2(history)

    #img = data[3]
    #img_array = keras.preprocessing.image.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0)  # Create a batch

    #predictions = model.predict(img_array)
    #score = tf.nn.softmax(predictions[0])

    #print(
     #   "This image most likely belongs to {} with a {:.2f} percent confidence."
      #      .format(labels[np.argmax(score)], 100 * np.max(score))
    #)


def plot1(history):
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
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
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    main()
