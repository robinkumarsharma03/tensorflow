import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt


celsius_q = np.array([-40,10,0,8,15,22,38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degree Fahrenhet".format(c, fahrenheit_a[i]))

la = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([la])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=False)
print("Finisehd training the model")

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

print(model.predict([100.0]))
print("These are the layer variables: {}".format(la.get_weights()))
print('It works')
