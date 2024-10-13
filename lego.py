import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import ntpath
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from imgaug import augmenters as iaa


basepath = "/kaggle/input/lego-brick-images/LEGO brick images v1"
img_file_paths = []
for _, i, _ in os.walk(basepath):
    for j in i:
        img_file_paths.append(os.path.join(basepath, j))

label_names = [
    "3022 Plate 2x2",
    "32123 half Bush",
    "3004 Brick 1x2",
    "6632 Technic Lever 3M",
    "3040 Roof Tile 1x2x45deg",
    "3069 Flat Tile 1x2",
    "3673 Peg 2M",
    "3713 Bush for Cross Axle",
    "3023 Plate 1x2",
    "18651 Cross Axle 2M with Snap friction",
    "3024 Plate 1x1",
    "3794 Plate 1X2 with 1 Knob",
    "3003 Brick 2x2",
    "11214 Bush 3M friction with Cross axle",
    "3005 Brick 1x1",
    "2357 Brick corner 1x2x2"
]

label_index = 0
img_labels = []
img_dataset = []
for img_file_path in img_file_paths:
    _, label = ntpath.split(img_file_path)
    print(label)
    for _,_,image_names in os.walk(img_file_path):
        for image_name in image_names:
            image = os.path.join(img_file_path, image_name)
            print(image)
            img_labels.append(label_index)
            image = cv2.imread(image)
            img_dataset.append(image)
    label_index += 1
print(label_index)
img_labels = np.array(img_labels)
img_dataset = np.array(img_dataset)

print(img_labels.shape)
print(img_dataset.shape)
plt.imshow(img_dataset[random.randint(0, img_dataset.shape[0] - 1), :, :, :])

num_of_samples = []
 
num_columns = 5
num_classes = 16
 
fig, axs = plt.subplots(nrows=num_classes, ncols=num_columns, figsize=(5, 20))
fig.tight_layout()

for col in range(num_columns):
    for row in range(num_classes):
        selected_images = img_dataset[img_labels==row]
        axs[row][col].imshow(selected_images[random.randint(0, len(selected_images) - 1), :, :, :], cmap=plt.get_cmap("gray"))
        axs[row][col].axis("off")
        if col == 2:
            axs[row][col].set_title(str(row) + " : " + label_names[row])
            num_of_samples.append(len(selected_images))


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")

X_train, X_valid_text, y_train, y_valid_text = train_test_split(img_dataset, img_labels, test_size=0.1, random_state=6)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_text, y_valid_text, test_size=0.3, random_state=6)


num_of_test_samples = []
num_of_valid_samples = []

for i in range(num_classes):
    test_selected = X_test[y_test==i]
    valid_selected = X_valid[y_valid==i]
    num_of_test_samples.append(len(test_selected))
    num_of_valid_samples.append(len(valid_selected))


print(num_of_valid_samples)
print(num_of_test_samples)
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
axs[0].bar(range(0, num_classes), num_of_valid_samples)
axs[0].set_title("Distribution of validation dataset")
axs[0].set_xlabel("Class number")
axs[0].set_ylabel("Number of images")
axs[1].bar(range(0, num_classes), num_of_test_samples)
axs[1].set_title("Distribution of test dataset")
axs[1].set_xlabel("Class number")
axs[1].set_ylabel("Number of images")
fig.tight_layout()


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

def preprocessing(img):
    img = cv2.resize(img, dsize=(96, 96), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
fig.tight_layout()
axs[0].imshow(X_train[100], cmap=plt.get_cmap("gray"))
axs[0].set_title("Original Image")
axs[0].axis("off")
axs[1].imshow(preprocessing(X_train[100]), cmap=plt.get_cmap("gray"))
axs[1].set_title("Preprocessed Image")
axs[1].axis("off")


def pan(image):
    pan = iaa.Affine(translate_percent= {"x" : (-0.05, 0.05), "y": (-0.05, 0.05)})
    image = pan.augment_image(image)
    return image

def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.2))
    image = zoom.augment_image(image)
    return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.5, 1.3))
    image = brightness.augment_image(image)
    return image

def img_random_flip(image):
    image = cv2.flip(image, 1)
    return image


def random_augmentation(image):

    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image = img_random_flip(image)
    
    return image


ncol = 2
nrow = 10
 
fig, axs = plt.subplots(nrow, ncol, figsize=(8, 20))
fig.tight_layout()
 
for i in range(10):
    randnum = random.randint(0, len(img_dataset) - 1)
    original_image = img_dataset[randnum]
    
    augmented_image = random_augmentation(original_image)
    
    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
  
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")

X_train = np.array(list(map(preprocessing, X_train)))
X_valid = np.array(list(map(preprocessing, X_valid)))
X_test = np.array(list(map(preprocessing, X_test)))
print(X_test.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_test.shape)

X_train = X_train / 255
X_test = X_test / 255
X_valid = X_valid / 255

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

def batch_generator(images, labels, batch_size, istraining):
  
    while True: 
        batch_img = []
        label_img = []
        for i in range(batch_size):
            random_index = random.randint(0, len(images) - 1)
            if istraining:
                img = random_augmentation(images[random_index])
                label = labels[random_index]
            else:
                img = images[random_index]
                label = labels[random_index]
            batch_img.append(img)
            label_img.append(label)
        yield (batch_img, label_img)

def LeNet_Model():
    model = Sequential()
    model.add(Conv2D(75, kernel_size=(5, 5), input_shape=(96, 96, 1), activation="relu"))
    model.add(Conv2D(75, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(45, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(45, kernel_size=(3, 3), activation="relu"))  
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 750, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units = 375, activation="relu"))
    model.add(Dense(units = num_classes, activation="softmax"))
    model.compile(Adam(lr = 0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = LeNet_Model()
print(model.summary())

history = model.fit_generator(
    batch_generator(X_train, y_train, 10, 1),
    steps_per_epoch=30,
    epochs=12,
    validation_data=batch_generator(X_valid, y_valid, 10, 0),
    validation_steps=20,
    verbose=1,
    shuffle = 1
)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
fig.tight_layout()
axs[0].plot(history.history["loss"])
axs[0].plot(history.history["val_loss"])
axs[0].legend(["loss", "Val_loss"])
axs[0].set_title("loss")
axs[0].set_xlabel("epoch")
axs[1].plot(history.history["accuracy"])
axs[1].plot(history.history["val_accuracy"])
axs[1].legend(["accuracy", "val_accuracy"])
axs[1].set_title("accuracy")
axs[1].set_xlabel("epoch")


score = model.evaluate(X_test, y_test, verbose = 1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
