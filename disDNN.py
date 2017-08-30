#-*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import time
import os
from DNNmodel import DNN

#get the optimizer
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

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GRPC_VERBOSITY_LEVEL']='DEBUG'

# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
#print(parameter_servers);
workers = sys.argv[2].split(',')
n_Workers = len(workers)
#print(workers);
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
learning_rate = 0.0005
logs_path = "/tmp/mnist/1"
targted_accuracy = FLAGS.targted_accuracy
Optimizer = FLAGS.optimizer

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/root/DMLcode/MNIST_data', one_hot=True)

if FLAGS.job_name == "ps":
#    print ("Launching a parameter server\n")
    server.join()
elif FLAGS.job_name == "worker":
#    print ("Launching a worker\n")
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	# count the number of updates
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)
	
	# input images
	with tf.name_scope('input'):
	# None -> batch size can be any size, 784 -> flattened mnist image
	    x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
	    # target 10 output classes
	    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

	with tf.name_scope('write_file'):
	    content = tf.placeholder(tf.string)
	    writer = tf.write_file("ex_re.csv", content)
	tf.set_random_seed(1)

	# Build the graph for the DNN
	y_con,keep_prob = DNN(x)
	
	# specify cost function
	with tf.name_scope('cross_entropy'):
	    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_con))
	
	# specify optimizer
	with tf.name_scope('train'):
	    grad_op = get_optimizer( Optimizer, learning_rate)
	    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
	
	#init_token_op = grad_op.get_init_tokens_op()
	#chief_queue_runner = rep_op.get_chief_queue_runner()
	#changed by andy
	
	with tf.name_scope('Accuracy'):
	# accuracy
	    correct_prediction = tf.equal(tf.argmax(y_con,1), tf.argmax(y_,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# create a summary for our cost and accuracy
	init_op = tf.global_variables_initializer()
	variables_check_op=tf.report_uninitialized_variables()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
				global_step=global_step,
				init_op=init_op)
    state = False
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
	while(not state):
            uninitalized_variables=sess.run(variables_check_op)
	    if(len(uninitalized_variables.shape) == 1):
		state = True
	# create log writer object (this will log on every machine)
	#writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	# perform training cycles
	start_time = time.time()
	step = 0
	final_accuracy = 0
	begin_time = time.time()
	batch_count = int(mnist.train.num_examples/batch_size)
	for i in range(batch_count*10):
	    batch_x, batch_y = mnist.train.next_batch(batch_size)
				
	    # perform the operations we defined earlier on batch
	    _, cost, step = sess.run([train_op, cross_entropy, global_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
	    final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	    if (final_accuracy > targted_accuracy):
		break
	    #writer.add_summary(summary, step)
	    print("Step: %d," % (step+1), 
			" Accuracy: %.4f," % sess.run(accuracy, feed_dict = {x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}), 
			" Batch: %3d of %3d," % (i+1, batch_count), 
			" Cost: %.4f," % cost, 
			" Time: %3.2fms" % float(time.time()-begin_time))
	#index, sum_step, total_time, cost, final_accuracy
	re = str(n_PS) + '-' + str(n_Workers) + '-' + str(FLAGS.task_index) + ',' + str(step) + ',' + str(float(time.time()-begin_time)) + ',' + str(cost) + ',' + str(final_accuracy)
	sess.run(writer, feed_dict = {content: re})
    sv.stop 
