from filenames import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
input_feature = np.genfromtxt(data_path + norm_express_file, delimiter=',')
print(input_feature.shape)


# Set parameters for learning
learning_rate = 0.01
num_steps = 200
batch_size = 16 

# Set parameters for display
display_step = 10

# Set parameters for network
num_input = 524
num_hidden_1 = 256
num_hidden_2 = 128

X = tf.placeholder("float", [None, num_input], name="input")

# Hidden layer settings
weights = {
    "encoder_h1": tf.Variable(tf.random_normal([num_input, num_hidden_1]), name="encoder_weights1"),
    "encoder_h2": tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]), name="encoder_weights2"),
    "decoder_h1": tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1]), name="decoder_weights1"),
    "decoder_h2": tf.Variable(tf.random_normal([num_hidden_1, num_input]), name="decoder_weights2"),
}

biases = {
    "encoder_b1": tf.Variable(tf.random_normal([num_hidden_1]), name="encoder_biases1"),
    "encoder_b2": tf.Variable(tf.random_normal([num_hidden_2]), name="encoder_biases2"),
    "decoder_b1": tf.Variable(tf.random_normal([num_hidden_1]), name="decoder_biases1"),
    "decoder_b2": tf.Variable(tf.random_normal([num_input]), name="decoder_biases2"),
}

# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']), name="encoder_layer1")
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']), name="encoder_layer2")
    return layer_2

# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), name="decoder_layer1")
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']), name="decoder_layer2")
    return layer_2

# Construct the model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pre = decoder_op
y = X

loss = tf.reduce_mean(tf.pow(y - y_pre, 2), name="mean_square_loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(loss)

init = tf.global_variables_initializer()

# Run network model, start traning process
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    sess.run(init)
    total_batch = int(input_feature.shape[0]/batch_size)

    for step in range(num_steps):
        for i in range(total_batch):
            batch_x = input_feature[i*batch_size : (i+1)*batch_size]
            _, l = sess.run([optimizer, loss], feed_dict={X:batch_x})
        if (step+1) % display_step == 0 or step == 0:
            print("Step %d: Minimize loss %f" % (step+1, l))

