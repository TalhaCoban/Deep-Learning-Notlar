import numpy as np 
import tensorflow.keras
#sequential model as per the offical documentation is a linear stack of letters
#Neural networks are actually organized in layers containing interconnected noted
from tensorflow.keras.models import Sequential
#every node in the layer is connected to every node in the preceding layer
#Dense will be used to construct densely-connected neural networks layers
from tensorflow.keras.layers import Dense
#the Adam optimizer item is one of many optimization algorithms
#Whats distinctive about the adam optimizer is its adaptive learning method 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 
plt.scatter(X[:n_pts,0], X[:n_pts,1], color = "r")
plt.scatter(X[n_pts:,0], X[n_pts:,1], color = "g")


model = Sequential()
"""
model.add(Dense(units: equal one will have a single output for our perceptron,
                input_shape: since this is the first layer being creadted we must also define a number of input notes,
                activation: defines the activation function used in the output layer nodes of perceptron
                ))
"""
model.add(Dense(units = 1, input_shape = (2,), activation = "sigmoid"))
"""
having specified the layers of our network to compile a keras model we need to make use of an optimizer
(bir keras modelini derlemek için ağımızın katmanlarını belirttikten sonra bir iyileştirici kullanmamız gerekiyor)
"""#gradient = (points.T * (p - y)) * (alpha / m)
   #line_parameters = line_parameters - gradient
"""the problem with taking this derivative is it is too expensive since we were essentially just multiplying a scaler time the coordinates of all the points
what if we were deaaling with a million points or 100 million points
the fact that we were looking at all the training examples at the same time is ehy we refer to this form of gradient descent as a batch gradient descent
A more efficient algorihm that better scale to large datasets is the """#stochastic gradient descent
"""which would be much more computationally efficiet than vanilla gradient descent. Why:
Well as we just saw Batch gradient descent computes the gradinet to using the entire datasets.
the stochastic gradient descent computed though a single sample
for the sake of moving on to the code, the adam optimization algorithm is a combination of two other extensions a stochastic gradient descent notably Adagrad and RMSprop and is a very efficient stochastic optimization method in updating the weights of our network

adam = Adam(lr : learning rate)
"""
adam = Adam(lr = 0.1)
"""
now before training our model to classify this data we need to configure the learning process which is done via the compile method
İnside of compile, we need to specifiy what kind of optimizer we were going to use, what kind of error funtion and tend to cross entropy and the metrics which is a function that you use to judge the performing of our model

model.compile(the optimizer such as Adam,
              loss: will equal some form of cross entropy which is the loss function that will help determine the error.
              metrics: is very similar to a loss function. however unlike the error function, whose results as we saw are constantly back propagate it to minimize the error our model
    the result from evaluating a metric are not used to train the model but simply to judge the performance at every epoch which is going to equal a list of functions)
"""
model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])
"""
This is the function that we use to start treating our perceptron. To start training and model that perfectly classify as our data
model.fit(x : x,
          y = y,
          verbose=1: but that will do is siply display a progress bar of information relating to the performance of our model at each epoch.
      if it is equal to the 0, it simply wouldnt print anything and we wouldnt know whats going on
      an epoch simply refers to whatever it iterates over the entire dataset of points and labels to try and seperate our data in discrete class based on their assigned labels. So everytime it iterates over the entire dataset of points that is an epoch)
          batch_size = one epoch is so big to feed to the computer all at ones. so we need to divede it into several smaller batches
      we want to make sure about size sufficiently big that way doesnt take forever to run our code.   
      we have 1000 datapoints with batch size of 50 it will take 20 iterations to compilete to one epoch and at each iterations its going to update the weigths of the neural netwok minimizing its error
          epoch: if we simply specify one epoch , that leads to the underfitting. As the number of epochs increases the more times its able to update the weights of our network. We dont want to pass in too many epochs since that can lead to overfitting   
          shuffle: keeps minimizing the error it will tent to get stuckin a local minimum of some sort rahther than absolute minimum
"""
h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs = 50, shuffle="True",)

plt.plot(h.history["accuracy"])
plt.title("accuracy")
plt.xlabel("epoch")
plt.legend(["accuracy"])


plt.plot(h.history["loss"])
plt.title("loss")
plt.xlabel("epoch")
plt.legend(["loss"])
plt.show()

def plot_decision_boundary(X,y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:, 1]) + 1)
    #masgrid function does is it allows us to return coordinate to matrices from the input of coordinate vectors
    xx , yy = np.meshgrid(x_span,y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape) 
    plt.contourf(xx, yy, z)

plot_decision_boundary(X,y,model)
x = 7.5
y = 5
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="yellow")
print("prediction is: ",prediction)

