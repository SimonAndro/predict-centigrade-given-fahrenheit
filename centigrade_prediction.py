"""
A dumy AI model for predicting temperature in degrees centigrade
given degrees fahrenheit 
"""

#
#import necessary modules
#
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

#
# create  a dumy dataset
#
#centigrade = (fahrenheit - 32)*(5/9) # conversion formula
sample_size = 1000
test_size = 0.25

# train set
x_train = np.random.uniform(1,100,(int(sample_size*(1-test_size)),1)) # generate random Fahrenheit values between 1 and 100
y_train = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_train]) # calculate corresponding Centigrade values

# test set
x_test = np.random.uniform(1,100,(int(sample_size*test_size),1)) # generate random Fahrenheit values between 1 and 100
y_test = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_test]) # calculate corresponding Centigrade values

#
# build the model
#
def build_model():
    """
    build a sequential dense neural network
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(1,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#
# model training
#
num_epochs = 20
model = build_model()
history = model.fit(x_train, y_train,
                        epochs=num_epochs, batch_size=8, verbose=1)
print(history.history)                        

#
# model evaluation
#
eval_mse, eval_mae = model.evaluate(x_test, y_test, verbose=1)
print("MSE=%s, MAE=%s" % (eval_mse, eval_mae))

#
# application of the trained model, predicting values
#
x_new = np.random.uniform(1,100,(50,1)) # generate random Fahrenheit values between 1 and 100
y_new = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_new]) # calculate corresponding Centigrade values

y_new_predicted = model.predict(x_new)
# show the inputs and predicted outputs
for i in range(len(y_new_predicted)):
	print("Actual=%s, Predicted=%s" % (y_new[i], y_new_predicted[i]))