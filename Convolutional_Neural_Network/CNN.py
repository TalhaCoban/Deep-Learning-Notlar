
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
#model class API is an increadible useful tool as it allows us to define a model
#model class API allows usto to instantiate layers from pre-trained models effectively allowing us to reuse section of previously trained models
#we are going to take advantages of this abiliy to helps us visualize the outputs from our 2Conv layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
#These two convolutional functions allows us to create convolutional and pooling layers within our network respectively
#our model have 2 convolutional layers and 2 pooling layers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.layers import Dropout
import random
import requests
from PIL import Image
import cv2


np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."

num_of_samples=[]
 
cols = 5
num_classes = 10
 
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()
 
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

X_train = X_train / 255
X_test = X_test / 255

#Define the leNet function
def LeNet_model():
    model = Sequential()
    """

    model.add(Conv2D(filters: how many filters our layer is going to have
                     kernel_size: each filter that is 5x5 etc
                     input_shape : as this is the wery first layer of out layer we need to spcify the shape of our input data
                     activation:
                     strides: kernel step one envolved on the image, how much a kernel translated
                     padding: simply ensures that the output size remains the same as the input size. to do so we add to pick so thick layers of padding with each pixels value equals to 0
                it allows to extract low level features and thus by keeping all the information at the borders, this tends to improve performance
                     padding = "valid" or "casual or "same
                we are not use padding for our network because we are not interested in the outer edges of our image
                     )
            )
    """
    #after above process our the shaphe of our convoluated image is going to be 24x24x30
    model.add(Conv2D(filters=30, kernel_size=(5,5), input_shape = (28,28,1), activation="relu"))
    """
    model.add(MaxPooling2D(pool_size : ))
    """
    model.add(MaxPooling2D(pool_size=(2,2)))
    #after above process our the shaphe of our convoluated image is going to be 12x12x30
    model.add(Conv2D(filters = 15, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    """
    Flatten Layer will help us to find our data in order to format it properly for it go in the fully connected layer
    we had to flat each image to be one dimensional
    """
    model.add(Flatten())
    model.add(Dense(units=500, activation="relu"))
    """
    typically the droput layer is used in between layers that have a high number of parameters because these high parameter layers are more likely to overfit and memorize the training datasets
    model.add(Dropout(rate: a fraction rate there is refers to the amount of input nodes that the dropout layer drops during each update with zero refer to when no notes are dropped and one referring to all input nodes are dropped)
    """
    model.add(Dropout(rate = 0.5))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(Adam(learning_rate=0.01), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model


model = LeNet_model()
print(model.summary())

history = model.fit(x = X_train, y = y_train, epochs = 10, validation_split=0.1, batch_size=600, verbose = 1, shuffle = 1)


model.save("model_2.h5")


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


url = 'https://printables.space/files/uploads/download-and-print/large-printable-numbers/3-a4-1200x1697.jpg'
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
image = image.reshape(1,28,28,1)

prediction = model.predict_classes(image)
print("predicted digit:", str(prediction[0]))

score = model.evaluate(X_test, y_test, verbose = 1)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

"""
layer1 = Model(
  first argument defines all the inputs into our network
  second argument defines output that we want from the model
)
"""
layer1 = Model(inputs = model.layers[0].input, outputs = model.layers[0].output)
layer2 = Model(inputs = model.layers[0].input, outputs = model.layers[2].output)

visual_layer1, visual_layer2 = layer1.predict(image), layer2.predict(image)
print(visual_layer1.shape)
print(visual_layer2.shape)

plt.figure(figsize=(10,6))
for i in range(30):
    plt.subplot(6,5,i+1)
    plt.imshow(visual_layer1[0 ,: , : , i], cmap = plt.get_cmap("jet"))
    plt.axis("off")
plt.show()

plt.figure(figsize=(10,6))
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(visual_layer2[0 ,: , : , i], cmap = plt.get_cmap("jet"))
    plt.axis("off")
plt.show()

