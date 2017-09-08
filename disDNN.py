#-*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import time
import os
from DNNmodel import DNN
from tensorflow.examples.tutorials.mnist import input_data

#get the optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GRPC_VERBOSITY_LEVEL']='DEBUG'

def get_optimizer(optimizer, learning_rate):
    if optimizer == "SGD":
	return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "Adadelta":
	return  tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == "Adagrad":
	return  tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == "Ftrl":
        return  tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == "Adam":
	return  tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "Momentum":
	return  tf.train.MomentumOptimizer(learning_rate)
    elif optimizer == "RMSProp":
	return  tf.train.RMSProp(learning_rate)
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(Shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.constant(0.0, shape = Shape)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
workers = sys.argv[2].split(',')
n_Workers = len(workers)
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_float("targted_accuracy", 0.5, "targted accuracy of model")
tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)
	
# config
batch_size = 100
learning_rate = 0.0001
targted_accuracy = FLAGS.targted_accuracy
Optimizer = FLAGS.optimizer

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	#More to come on is_chief...
        is_chief = FLAGS.task_index == 0
	# count the number of updates
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
	# load mnist data set
	mnist = input_data.read_data_sets('./code/disDNN/MNIST_data', one_hot=True)
	# input images
	with tf.name_scope('input'):
	    # None -> batch size can be any size, 784 -> flattened mnist image
	    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
	    # target 10 output classes
	    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

	with tf.name_scope('reshape'):
    	    x_image = tf.reshape(x, [-1, 28, 28, 1])

  	with tf.name_scope('conv1'):
    	    W_conv1 = weight_variable([5, 5, 1, 32])
	    b_conv1 = bias_variable([32])
	    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  
  	with tf.name_scope('pool1'):
    	    h_pool1 = max_pool_2x2(h_conv1)

  	with tf.name_scope('conv2'):
    	    W_conv2 = weight_variable([5, 5, 32, 64]) 
    	    b_conv2 = bias_variable([64])
    	    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  	with tf.name_scope('pool2'):
    	    h_pool2 = max_pool_2x2(h_conv2)

  	with tf.name_scope('fc1'):
    	    W_fc1 = weight_variable([7 * 7 * 64, 1024]) 
    	    b_fc1 = bias_variable([1024])
    	    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    	    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  	with tf.name_scope('dropout'):
    	    keep_prob = tf.placeholder(tf.float32)
    	    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  	with tf.name_scope('softmax'):
    	    W_fc2 = weight_variable([1024, 10])
    	    b_fc2 = bias_variable([10])

  	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# specify cost function
	with tf.name_scope('cross_entropy'):
	    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	# specify optimizer
	with tf.name_scope('train'):
	    grad_op = get_optimizer( Optimizer, learning_rate)
	    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
	# accuracy
	with tf.name_scope('Accuracy'):
	    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	saver = tf.train.Saver({"W_conv1": W_conv1,
                          "b_conv1": b_conv1,
                          "W_conv2": W_conv2,
                          "b_conv2": b_conv2,                                     
			  "W_fc1": W_fc1,                                                                         
			  "b_fc1": b_fc1,
                          "W_fc2": W_fc2, 
			  "b_fc2": b_fc2})
	variables_check_op=tf.report_uninitialized_variables()
    	sess_config = tf.ConfigProto(
        	allow_soft_placement=True,
        	log_device_placement=False,
        	device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    sv = tf.train.Supervisor(is_chief=is_chief,
			     saver=saver,
                             global_step=global_step,
                             save_model_secs=600)
    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False
    with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
	saver.restore(sess, "code/disDNN/savemodel/initialize_wb.ckpt")
	while(not state):
            uninitalized_variables=sess.run(variables_check_op)
	    if(len(uninitalized_variables.shape) == 1):
		state = True
	start_time = time.time()
	step = 0
	cost = 0
	final_accuracy = 0
	begin_time = time.time()
	while (not sv.should_stop()) and (step < 550):
	    batch_x, batch_y = mnist.train.next_batch(batch_size)
	    _, cost, step = sess.run([train_op, cross_entropy, global_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
	    #final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	    #if (final_accuracy > targted_accuracy):
	    #	break
	    print("Step: %d," % (step+1), 
			" Accuracy: %.4f," % final_accuracy,
			" Loss: %f" % cost,
			" Time: %fs" % float(time.time()-begin_time))
	    begin_time = time.time()
	    
	#index, sum_step, total_time, cost, final_accuracy
	    
	final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	re = str(n_PS) + '-' + str(n_Workers) + '-' + str(FLAGS.task_index) + ',' + str(step) + ',' + str(float(time.time()-start_time)) + ',' + str(cost) + ',' + str(final_accuracy)
        writer = open("re_2_"+Optimizer+".csv", "a+")
        writer.write(re+"\r\n")
        writer.close()
    sv.stop 
