import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import requests
import cv2
from PIL import Image


np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equual to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equual to the number of labels"
assert(X_train.shape[1:0] == y_train.shape[1:0]), "The dimensions of the images are not 28x28"
assert(X_test.shape[1:0] == y_test.shape[1:0]), "The dimensions of the images are not 28x28"

number_of_sample = []

cols = 5
number_classes = 10 
fig,axs = plt.subplots(nrows = number_classes, ncols = cols, figsize = (6,10))
fig.tight_layout()
for i in range(cols):
    for j in range(number_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)), :, :], cmap = plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            number_of_sample.append(len(x_selected))


print("number_of_sample: ", number_of_sample)
plt.figure(figsize=(12,4))
plt.bar(range(0,number_classes), number_of_sample)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of İmages")
plt.show()

#one hot coding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#normalization
X_train = X_train / 255
X_test = X_test / 255

number_pixels = 784
X_train = X_train.reshape(X_train.shape[0], number_pixels)
X_test = X_test.reshape(X_test.shape[0], number_pixels)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ",X_test.shape)


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim = number_pixels, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(number_classes, activation = "softmax"))
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics = ["accuracy"])
    return model

model = create_model()
print(model.summary())

"""
model.fit(X_train, 
          y_train,
          validation_split = to measure how well its able to generalize to it)
"""
h = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose=1, shuffle = 1)


model.save("model_1.h5")
           

plt.plot(h.history["loss"])
plt.plot(h.history["val_loss"])
plt.legend(["loss", "Val_loss"])
plt.title("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.legend(["accuracy", "Val_accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.show()

score = model.evaluate(X_test, y_test, verbose = 1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])


url = "https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png"
response = requests.get(url, stream = True)
print(response)
img = Image.open(response.raw)
plt.imshow(img)

img_array = np.asarray(img)
print(img_array.shape)
resized  = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
print(image.shape)
plt.imshow(image, cmap = plt.get_cmap("gray"))

image = image / 255
image = image.reshape(1,784)

prediction = model.predict_classes(image)
print("predicted digit:", str(prediction[0]))
