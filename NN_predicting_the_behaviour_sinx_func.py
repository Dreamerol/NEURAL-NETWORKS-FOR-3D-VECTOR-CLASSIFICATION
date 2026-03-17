import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_training_points = np.linspace(-1,1,200)
x_test_points = np.arange(-1,1,0.15)

def f(x):
  return np.sin(x)

y_output_training_points = np.array([f(x) for x in x_training_points])
y_output_test_points = np.array([f(x) for x in x_test_points])

model = tf.keras.Sequential([
      tf.keras.layers.Dense(64,activation='tanh', input_shape=(1,)),
      tf.keras.layers.Dense(64,activation='tanh'),
      tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

#mae - mean absolute error
#mse - mean squared error

model.fit(x_training_points, y_output_training_points,epochs=100)
predictions = model.predict(x_test_points)

plt.plot(x_test_points, predictions)
plt.plot(x_test_points, f(x_test_points))
