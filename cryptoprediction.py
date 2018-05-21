# Imports: numPy, Pandas, TensorFlow, scikit-learn, matplotlib
# Using Python 3.x
# ----------------------------------
# Author: "Antony Tokarr" (Manokhin)
# Spring 2018
# COGS 298 - Deep Learning
# Prof. Josh de Leeuw
# ----------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plot

# Read our dataset and drop the time variable.
# data_cryptocurrencies is a dataset of 1832 rows and 15 columns
# (I tried using .Shape[] but was getting out-of-bounds/NaN errors)
dataset = pd.read_csv('data_cryptocurrencies.csv', encoding = "ISO-8859-1", 
    engine='python')
dataset = dataset.drop(['TIME'], 1)
rows = 1832 
columns = 15
dataset = dataset.values # Represent our dataset as an array (numpy)

# Prepare the training and testing sections of our data.
# training_data is composed of the first 7/10s of the total dataset.
training_first = 0
training_last = int(np.floor(0.70*rows))
training_data = dataset[np.arange(training_first, training_last), :]
# testing_data is composed of the remaining 3/10s of the total dataset.
testing_first = training_last + 1
testing_last = rows
testing_data = dataset[np.arange(testing_first, testing_last), :]

# Use sklearn.MinMaxScaler to scale/convert training and testing data.
# Done for purposes of the tanh neurons of the network: [-1, 1]
# ****
# tanh is rescaled type of sigmoid with output [-1, 1]
# rather than the typical [0, 1] that sigmoid has.
tanh_convert = MinMaxScaler(feature_range=(-1, 1))
tanh_convert.fit(training_data)
training_data = tanh_convert.transform(training_data)
testing_data = tanh_convert.transform(testing_data)
training_x = training_data[:, 1:]
training_y = training_data[:, 0]
testing_x = testing_data[:, 1:]
testing_y = testing_data[:, 0]

# Total number of cryptocurrencies in our training dataset.
num_currencies = training_x.shape[1]

# Number of neurons per layer.  First layer 1024, second layer 512, etc.
numTahnNeuronsLayer1 = 1024
numTahnNeuronsLayer2 = 512
numTahnNeuronsLayer3 = 256
numTahnNeuronsLayer4 = 128

# Setup tensorflow and create placeholders with dataset to be filled later.
# Initalize sigma, biases, and weights.
session = tf.InteractiveSession()
# 2-dimensional input vector
inputsX_2D = tf.placeholder(dtype=tf.float32, shape=[None, num_currencies])
# 1-dimensional output vector
outputsY_1D = tf.placeholder(dtype=tf.float32, shape=[None])
initialize_sigma = 1
initialize_biases = tf.zeros_initializer()
initialize_weights = tf.variance_scaling_initializer(mode="fan_avg", 
    distribution="uniform", scale=initialize_sigma)

# ----
# Create the variables for each layer's biases and weights.
# First Layer:
biases_first_layer = tf.Variable(initialize_biases([numTahnNeuronsLayer1]))
weights_first_layer = tf.Variable(initialize_weights([num_currencies, 
    numTahnNeuronsLayer1]))
# Second Layer:
biases_second_layer = tf.Variable(initialize_biases([numTahnNeuronsLayer2]))
weights_second_layer = tf.Variable(initialize_weights([numTahnNeuronsLayer1, 
    numTahnNeuronsLayer2]))
# Third Layer:
biases_third_layer = tf.Variable(initialize_biases([numTahnNeuronsLayer3]))
weights_third_layer = tf.Variable(initialize_weights([numTahnNeuronsLayer2, 
    numTahnNeuronsLayer3]))
# Fourth Layer:
biases_fourth_layer = tf.Variable(initialize_biases([numTahnNeuronsLayer4]))
weights_fourth_layer = tf.Variable(initialize_weights([numTahnNeuronsLayer3, 
    numTahnNeuronsLayer4]))
# Output Layer:
weights_output_layer = tf.Variable(initialize_weights([numTahnNeuronsLayer4, 1]))
biases_output_layer = tf.Variable(initialize_biases([1]))
# ----

# Setup the architecture of the network by using the placeholders (our dataset)
# and the biases and weights.
first_layer = tf.nn.relu(tf.add(tf.matmul(inputsX_2D, 
    weights_first_layer), biases_first_layer))
second_layer = tf.nn.relu(tf.add(tf.matmul(first_layer, # first layer is used in next layer
    weights_second_layer), biases_second_layer))
third_layer = tf.nn.relu(tf.add(tf.matmul(second_layer, # And so on...
    weights_third_layer), biases_third_layer))
fourth_layer = tf.nn.relu(tf.add(tf.matmul(third_layer, # And so forth
    weights_fourth_layer), biases_fourth_layer))
output_layer = tf.transpose(tf.add(tf.matmul(fourth_layer,
    weights_output_layer), biases_output_layer))

# ----
# Our cost fxn is Mean Squared Error (MSE), compute the avg squared deviation of
# our predictions and our targets. 
Mean_Squared_Error = tf.reduce_mean(tf.squared_difference(output_layer, outputsY_1D))

# Create optimizer variable to calculate gradients to be 
# used for the biases and weights.
# ***
# Use tf's AdamOptimizer: "Adaptive Moment Estimation"
opt = tf.train.AdamOptimizer().minimize(Mean_Squared_Error)
# ----


# -------------------------------------------------------------------------------
# Run and Initialize Below
# -------------------------------------------------------------------------------
# Initialize the session and mean_squared_errors for training and testing data.
session.run(tf.global_variables_initializer())
mean_squared_error_training = []
mean_squared_error_testing = []

# Create our plot using matplotlib, a dynamic plot.
plot.ion()
newFigure = plot.figure()
newAxis = newFigure.add_subplot(111)
# Set up the lines
line1, = newAxis.plot(testing_y)
line2, = newAxis.plot(testing_y * 4) # Multiply by 4 to see total view.
plot.show()

# Now run the program!
numEpochs = 200 # **Feel free to change, 200 was a good number for me.**
batchSize = 256
# Go through the total number of epochs
for ne in range(numEpochs):

    # Using numpy, randomize the training data
    randomizeTrainingData = np.random.permutation(np.arange(len(training_y)))
    # We do this to prevent any biases from the sorted CSV file
    training_x = training_x[randomizeTrainingData]
    training_y = training_y[randomizeTrainingData]

    # Take random data samples from the training data
    # and feed it into the network.
    # Then run the optimizer on the newly created sample batch.
    for index in range(0, len(training_y) // batchSize):
        initial = index * batchSize
        miniBatchX = training_x[initial:initial + batchSize]
        miniBatchY = training_y[initial:initial + batchSize]
        session.run(opt, feed_dict={inputsX_2D: miniBatchX, outputsY_1D: miniBatchY})

        # For every fourth batch, display the neural network's progress 
        if np.mod(index, 25) == 0:
            # ----
            # Display mean squared error for training and testing:
            mean_squared_error_training.append(session.run(Mean_Squared_Error, 
                feed_dict={inputsX_2D: training_x, outputsY_1D: training_y}))
            mean_squared_error_testing.append(session.run(Mean_Squared_Error, 
                feed_dict={inputsX_2D: testing_x, outputsY_1D: testing_y}))
            print('MeanSquaredError Train: ', mean_squared_error_training[-1])
            print('MeanSquaredError  Test: ', mean_squared_error_testing[-1])
            # ----
            # Using matplotlib, plot the prediction:
            predicted = session.run(output_layer, feed_dict={inputsX_2D: testing_x})
            line2.set_ydata(predicted)
            plot.title('numEpochs ' + str(ne) + ', numBatch ' + str(index))
            plot.pause(0.0001) # ***FEEL FREE TO ADJUST FOR FASTER OR SLOWER VIEWING***

# Print the final mean squared error
mean_squared_error_final = session.run(Mean_Squared_Error, 
    feed_dict={inputsX_2D: testing_x, outputsY_1D: testing_y})
print("Final Mean Squared Error is:")
print(mean_squared_error_final)
# Got 256 one time... Should've saved it.

# Recording Final Mean Squared Results: Trial Runs
#  1: 387.100
#  2: 4374.7363
#  3: 1713.9575
#  4: 484.83093
#  5: 914.0567
#  6: 2014.0585
#  7: 7713.2705
#  8: 73.09535
#  9: 555.4124
# 10: 555.4557