import tensorflow as tf
from data_initializer import DataInitializer

N_INPUT = 3136
N_CLASSES = 33
CNN_DROPOUT = 0.75

learning_rate = 0.001
training_iters = 200
batch_size = 10
display_step = 10

DIR = "../data/categories"

weights = {
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	'wd1': tf.Variable(tf.random_normal([14 * 14 * 64, 1024])),
	'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32)

x = tf.reshape(x, shape=[-1, 56, 56, 1])
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

pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

data_init = DataInitializer()
for k in range(0, training_iters):
	# print "next batch"
	batch = data_init.next_batch(batch_size)
	# for i in batch:
	# 	for j, t in enumerate(i):
	# 		if t > 0.0:
	# 			print j
	# 			break


# init = tf.initialize_all_variables()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
# 	sess.run(init)
