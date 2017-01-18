import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

x = tf.placeholder("float", [None, 784]) #mnist data image of shape 28*28*784
y = tf.placeholder("float", [None, 10]) #0-9 digits recognition

#Create model
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

learning_rate = 0.1;
# Construction model
activation = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

training_epochs = 201;
batch_size = 20;
display_step = 20;


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0;
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    correct_prediction = tf.equl(tf.argmax(activation, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print(accuracy.eval({x: mninst.test.images, y: mnist.test.labels}))
