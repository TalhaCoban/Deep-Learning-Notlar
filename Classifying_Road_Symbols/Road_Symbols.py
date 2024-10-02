import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np.random.seed(1)

"""rb : reading the file in a binary format"""
with open("train.p", "rb") as f:
  train_data = pickle.load(f)
with open("valid.p", "rb") as f:
  val_data = pickle.load(f)
with open("test.p", "rb") as f:
  test_data = pickle.load(f)

print(train_data["features"].shape)
print(train_data["labels"].shape)
X_train, y_train = train_data["features"], train_data["labels"]
X_val, y_val = val_data["features"], val_data["labels"]
X_test, y_test = test_data["features"], test_data["labels"]

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equual to the number of labels"
assert(X_val.shape[0] == y_val.shape[0]), "The number of images is not equual to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equual to the number of labels"
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32"
assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32"
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32"

data = pd.read_csv("signnames.csv")

num_of_samples = []
 
cols = 5
num_classes = 43
 
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 40))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))
plt.show()

"""
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

#plt.imshow(X_train[999])
#plt.axis("off")
#print(X_train[999].shape)
#print(y_train[999])
"""


def grayScale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

"""
img = grayScale(X_train[999])
plt.imshow(img, cmap = plt.get_cmap("gray"))
plt.axis("off")
plt.show()
print(img.shape)
"""

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

"""
img = equalize(img)
plt.imshow(img, cmap = plt.get_cmap("gray"))
plt.axis("off")
plt.show()
print(img.shape)
"""

def preprocessing(img):
    img = grayScale(img)
    img = equalize(img)
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))
print(X_test.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_test.shape)


datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10.)
 
# for X_batch, y_batch in

batches_train = datagen.flow(X_train, y_train, batch_size = X_train.shape[0])
X_batch_train, y_batch_train = next(batches_train)
 
print("X_batch_train.shape: " ,X_batch_train.shape)
print("X_train.shape: " ,X_train.shape)
print("y_batch_train.shape: " ,y_batch_train.shape)
print("y_train.shape: " ,y_train.shape)
X_train = np.vstack((X_train, X_batch_train))
y_train = np.hstack((y_train, y_batch_train))
print("X_train.shape: " ,X_train.shape)
print("y_train.shape: " ,y_train.shape)

batches_val = datagen.flow(X_val, y_val, batch_size = X_val.shape[0])
X_batch_val, y_batch_val = next(batches_val)
 
print("X_batch_val.shape: " ,X_batch_val.shape)
print("X_val.shape: " , X_val.shape)
print("y_batch_val.shape: " ,y_batch_val.shape)
print("y_val.shape: " ,y_val.shape)
X_val = np.vstack((X_val, X_batch_val))
y_val = np.hstack((y_val, y_batch_val))
print("X_val.shape: " ,X_val.shape)
print("y_val.shape: " ,y_val.shape)

batches_test = datagen.flow(X_test, y_test, batch_size = X_test.shape[0])
X_batch_test, y_batch_test = next(batches_test)
 
print("X_batch_test.shape: " ,X_batch_test.shape)
print("X_test.shape: " ,X_test.shape)
print("y_batch_test.shape: " ,y_batch_test.shape)
print("y_test.shape: " ,y_test.shape)
X_test= np.vstack((X_test, X_batch_test))
y_test = np.hstack((y_test, y_batch_test))
print("X_test.shape: " ,X_test.shape)
print("y_test.shape: " ,y_test.shape)



X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


def LeNet_model():
    model = Sequential()
    model.add(Conv2D(70, (5, 5), input_shape = (32, 32, 1), activation="relu"))
    model.add(Conv2D(70, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(35, (3, 3), activation="relu"))
    model.add(Conv2D(35, (3, 3), activation="relu"))
    #model.add(Dropout(rate = 0.50))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 500, activation="relu"))
    model.add(Dropout(rate = 0.50))
    #model.add(Dense(units = 250, activation="relu"))
    model.add(Dense(units = num_classes, activation="softmax"))
    model.compile(Adam(lr = 0.0005), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

model = LeNet_model()
print(model.summary())

"""
history = model.fit_generator(
    datagen.flow(
        x = X_train, 
        y = y_train, 
        batch_size = 50
    ),
    steps_per_epoch = len(X_train) // 50,
    epochs = 10, 
    validation_data = (X_val, y_val),  
    shuffle=1
)
"""
history = model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = 400, epochs = 10, verbose = 1, shuffle = 1)


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "Val_loss"])
plt.title("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["accuracy", "Val_accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.show()

score = model.evaluate(X_test, y_test, verbose = 1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

#fetch image
     
img1 = cv2.imread("test/1.jpg")
img1 = np.asarray(img1)
img1 = cv2.resize(img1, (32, 32))
img1 = preprocessing(img1)
img1 = img1.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img1)) + "actual Class: 1")

img2 = cv2.imread("test/2.jpg")
img2 = np.asarray(img2)
img2 = cv2.resize(img2, (32, 32))
img2 = preprocessing(img2)
img2 = img2.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img2)) + "actual Class: 34")

img3 = cv2.imread("test/3.jpg")
img3 = np.asarray(img3)
img3 = cv2.resize(img3, (32, 32))
img3 = preprocessing(img3)
img3 = img3.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img3)) + "Actual Class: 23")

img4 = cv2.imread("test/4.jpg")
img4 = np.asarray(img4)
img4 = cv2.resize(img4, (32, 32))
img4 = preprocessing(img4)
img4 = img4.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img4)) + "actual Class: 13")


img5 = cv2.imread("test/5.jpg")
img5 = np.asarray(img5)
img5 = cv2.resize(img5, (32, 32))
img5 = preprocessing(img5)
img5 = img5.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img5)) + "Actual Class: 29")


img6 = cv2.imread("test/6.jpg")
img6 = np.asarray(img6)
img6 = cv2.resize(img6, (32, 32))
img6 = preprocessing(img6)
img6 = img6.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img6)) + "Actual Class: 4")


img7 = cv2.imread("test/7.jpg")
img7 = np.asarray(img7)
img7 = cv2.resize(img7, (32, 32))
img7 = preprocessing(img7)
img7 = img7.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img7)) + "Actual Class: 14")

img8 = cv2.imread("test/8.jpg")
img8 = np.asarray(img8)
img8 = cv2.resize(img8, (32, 32))
img8 = preprocessing(img8)
img8 = img8.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict_classes(img8)) + "Actual Class: 14")

