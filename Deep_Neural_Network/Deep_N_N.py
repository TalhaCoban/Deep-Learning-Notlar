import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


np.random.seed(0)

n_pts = 500
X, y = datasets.make_circles(n_samples = n_pts, random_state = 123, noise = 0.1, factor = 0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color = "b")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = "g")
plt.show()

model = Sequential()
"""
model.add(Dense(4: 4 nodes in hidden layer,
                input_shape = (2,),
                activation function = "sigmiod",
                ))
"""
model.add(Dense(4, input_shape = (2,), activation="sigmoid")) # we successfully defined our first layer and  the hidden layer
"""
we just have the final output layer to define before our deep neural network is complete
connect to onenode in the final layer
this is the final layer thats going to bethe output layer
"""
model.add(Dense(1, activation="sigmoid")) # the structure of our neural network is now complete
"""
we must now take our model and compile it and we are going to be using the Adam optimizer for our network
"""
model.compile(Adam(lr = 0.01), "binary_crossentropy", metrics = ["accuracy"])

"""
epoch simply refers to whenever it iterates over the entire dataset of points and labels to train and seperate our data and discrete classes based on their assigned labels
One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
Since one epoch is too big to feed to the computer at once we divide it in several smaller batches
Batch Size: Total number of training examples present in a single batch.
Iterations is the number of batches needed to complete one epoch 
Note: The number of batches is equal to number of iterations for one epoch.
at each iteration weights it will update the weights of the neural network to minimize the error
"""
#As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve
h = model.fit(x = X, y = y, verbose=1, batch_size = 20, epochs = 120, shuffle = "True")

plt.plot(h.history["accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.legend(["accuracy"])


plt.plot(h.history["loss"])
plt.title("loss")
plt.xlabel("epoch")
plt.legend(["loss"])

def plot_decision_boundary(X,y, model):
    x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:, 1]) + 0.25)
    #masgrid function does is it allows us to return coordinate to matrices from the input of coordinate vectors
    xx , yy = np.meshgrid(x_span,y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape) 
    plt.contourf(xx, yy, z)

plot_decision_boundary(X,y,model)
x = 0.2
y = 0.5
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
plt.show()
print("prediction is: ",prediction)

