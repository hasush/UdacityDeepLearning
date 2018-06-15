import tensorflow as tf

### Constant tensors. ###
# THe value of constant tensors never changes.

# Create tensor object.
hello_constant = tf.constant('Hello World!')

# 0-dimensional tensors.
stringTensor = tf.constant('String') # string tensor
intTensor0 = tf.constant(1234) # int32 tensor

# 1 dimensional int32 tensor.
intTensor1= tf.constant([123,456,789])

# 2 dimensional int32 tensor.
intTensor2 = tf.constant([ [1234,456,789], [222,333,444] ])

# Perform x/y - 1
asdf1 = tf.constant(10.0)
asdf2 = tf.constant(2.0)
asdf3 = tf.subtract(tf.divide(asdf1,asdf2), tf.constant(1.0))

### Place holders. ###
# variables allowing input from different datasets for different parameters.
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

asdfX = tf.placeholder(tf.float32)
asdfY  = tf.placeholder(tf.float32)
asdfZ = tf.subtract(tf.divide(asdfX, asdfY), tf.constant(1.0))



# The environment for running the tensor flow graph.
with tf.Session() as sess:

	# Run tf.constant operation in the sesion
	output = sess.run(hello_constant)
	print(output)

	# Use feed_dict with session.run() to set a placeholder value.
	output = sess.run(x, feed_dict={x:'Herro World'})
	print(output)
	print(x)

	# Set multiple items.
	output = sess.run((x,y,z) ,feed_dict={x: 'Test String', y:123, z:45.67})
	print(output)

	# Show the result of performing arthimetic operations outside of the session.
	print(sess.run(asdf3))

	# Show the result of performing arthimetic operations inside of the session.
	output = sess.run(asdfZ, feed_dict={asdfX:10.0, asdfY:2.0})
	print(output)
	print(asdfZ)

