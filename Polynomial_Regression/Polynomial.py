import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(11)

points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

model = Sequential()
model.add(Dense(500, input_dim = 1, activation = "sigmoid"))
model.add(Dense(30, activation= "sigmoid"))
model.add(Dense(1))
model.compile(Adam(lr = 0.01), loss = "mse")

model.fit(X, y, epochs = 50)

plt.scatter(X, y)
plt.plot(X, model.predict(X), "ro")
plt.show()

