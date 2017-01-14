import tensorflow as tf
sess = tf.Session()

a = tf.constant(1)
b = tf.constant(2)
c = a+b

print (c)
print (sess.run(c))
#everything is operation!


# place holder

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print ("Added result:%i" % sess.run(add, feed_dict={a:2, b:3}))
    print ("Muled result:%i" % sess.run(mul, feed_dict={a:2, b:3}))
