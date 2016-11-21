import tensorflow as tf
import vnist
import image_reader as ir

N_CLASSES = 33
CNN_DROPOUT = 0.7

learning_rate = 0.0005
training_iters = 200000
batch_size = 128
display_step = 10

DIR = "../data"
MODEL_PATH = "../data/conv_model.ckpt"

weights = {
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	'wd1': tf.Variable(tf.random_normal([ir.N_INPUT * 4, 2048])),
	'wd2': tf.Variable(tf.random_normal([2048, 1024])),
	'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([2048])),
	'bd2': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

x = tf.placeholder(tf.float32, [None, ir.N_INPUT])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)

x = tf.reshape(x, shape=[-1, ir.IMAGE_SIZE, ir.IMAGE_SIZE, 1])
conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, biases['bc1'])
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, biases['bc2'])
conv2 = tf.nn.relu(conv2)
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)

fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, keep_prob)

pred = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

vnist = vnist.read_data_sets(DIR)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < training_iters:
		batch_x, batch_y = vnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape([-1, ir.IMAGE_SIZE, ir.IMAGE_SIZE, 1])
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: CNN_DROPOUT})
		if step % display_step == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
			print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
				"{:.6f}".format(loss) + ", Training Accuracy= " + \
				"{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"
	save_path = saver.save(sess, MODEL_PATH)
	print "Model saved in file: %s" % save_path

	print "Testing Accuracy:", \
		sess.run(accuracy, feed_dict={
			x: vnist.test.images.reshape([-1, ir.IMAGE_SIZE, ir.IMAGE_SIZE, 1]),
			y: vnist.test.labels,
			keep_prob: 1.
		})

# with tf.Session() as sess:
# 	# sess.run(init)
# 	saver.restore(sess, MODEL_PATH)
# 	print "Testing Accuracy:", \
# 		accuracy.eval(feed_dict={
# 			x: vnist.test.images.reshape([-1, ir.IMAGE_SIZE, ir.IMAGE_SIZE, 1]),
# 			y: vnist.test.labels,
# 			keep_prob: 1.
# 		})
