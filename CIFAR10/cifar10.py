import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import cv2
import PIL
import random
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split


classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

random.seed(0)

(X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.1, random_state=6)


assert(X_train.shape[0] == y_train.shape[0]), "The number of train images is not equal to number of labels"
assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of train images are not 32 x 32 x 3"
assert(X_valid.shape[0] == y_valid.shape[0]), "The number of validation images is not equal to number of labels"
assert(X_valid.shape[1:] == (32, 32, 3)), "The dimensions of validation images are not 32 x 32 x 3"
assert(X_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to number of labels"
assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of test images are not 32 x 32 x 3"


num_of_samples = []

num_classes = 10
num_rows = 2

fig, axs = plt.subplots(nrows=num_rows, ncols=num_classes, figsize=(10, 4))

for j in range(num_rows):
    for i in range(num_classes):
        x_selected = X_train[y_train[:,0]==i]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :])
        axs[j][i].axis("off")
        if j == 0:
            axs[j][i].set_title(classes[i])
            num_of_samples.append(len(x_selected))
plt.show()

plt.figure(figsize=(10,4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of Classes")
plt.xlabel("Classes")
plt.ylabel("Number of Samples")
plt.show()


datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.0)

y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)
y_test = to_categorical(y_test, num_classes)

X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255


def VGG():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = VGG()

print("\n\n")
print(model.summary())
print("\n\n")

history = model.fit_generator(
    datagen.flow(
        x = X_train, 
        y = y_train, 
        batch_size = 100
    ),
    steps_per_epoch = len(X_train) // 100,
    epochs = 36,
    validation_data = (X_valid, y_valid),  
    shuffle=1,
    verbose=1
)


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.tight_layout()
ax[0].plot(history.history["loss"], color="blue", label="train set")
ax[0].plot(history.history["val_loss"], color="green", label="validation set")
ax[0].set_title("loss function")
ax[0].legend()
ax[1].plot(history.history["accuracy"], color="blue", label="train set")
ax[1].plot(history.history["val_accuracy"], color="green", label="validation set")
ax[1].set_title("accuracy function")
ax[1].legend()
plt.show()


test_images = []
for i in range(7):
    img_path = os.path.join("test_images/{}.jpg".format(str(i+1)))
    test_images.append(img_path)

fig, ax = plt.subplots(1, len(test_images), figsize = (12,4))
fig.tight_layout()

i = 0
for image in test_images:
    image = cv2.imread(image)
    img = np.asarray(image)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 3)
    print("predicted class: "+ classes[model.predict_classes(img)[0]] + " ----  actual Class: cat")
    ax[i].imshow(image)
    ax[i].set_title("{}(cat)".format(classes[model.predict_classes(img)[0]]), color = ("green" if model.predict_classes(img)[0] == "3" else "red"))
    i += 1
plt.show()

print("\n\n")
score = model.evaluate(X_test, y_test, verbose = 1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])


sorgu = input("model kaydedilsin mi? [y] [n]")
if sorgu == "y":   
    model.save('model.h5')

