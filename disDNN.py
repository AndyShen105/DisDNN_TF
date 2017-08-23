#-*-coding:UTF-8-*-
'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 
'''

from __future__ import print_function

import tensorflow as tf
import sys
import time
import os

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

	    # model parameters will change during training so we use tf.Variable
	tf.set_random_seed(1)
	with tf.name_scope("weights"):
	    W1 = tf.Variable(tf.random_normal([784, 100]))
	    W2 = tf.Variable(tf.random_normal([100, 50]))
	    W3 = tf.Variable(tf.random_normal([50, 25]))
	    W4 = tf.Variable(tf.random_normal([25, 10]))

	# bias
	with tf.name_scope("biases"):
	    b1 = tf.Variable(tf.zeros([100]))
	    b2 = tf.Variable(tf.zeros([50]))
	    b3 = tf.Variable(tf.zeros([25]))
	    b4 = tf.Variable(tf.zeros([10]))	
	# implement model
	with tf.name_scope("softmax"):
	    # y is our prediction
	    z2 = tf.add(tf.matmul(x, W1), b1)
	    a2 = tf.nn.sigmoid(z2)
	    z3 = tf.add(tf.matmul(a2, W2), b2)
	    a3 = tf.nn.sigmoid(z3)
	    z4 = tf.add(tf.matmul(a3, W3), b3)
	    a4 = tf.nn.sigmoid(z4)
	    z5 = tf.add(tf.matmul(a4, W4), b4)
	    y  = tf.nn.softmax(z5)

	# specify cost function
	with tf.name_scope('cross_entropy'):
	# this is our cost
	    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	# specify optimizer
	with tf.name_scope('train'):
	    # optimizer is an "operation" which we can execute in a session
	    grad_op = get_optimizer( Optimizer, learning_rate)
	    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
		
	#changed by andy
	#init_token_op = grad_op.get_init_tokens_op()
	#chief_queue_runner = rep_op.get_chief_queue_runner()
	#changed by andy
	
	with tf.name_scope('Accuracy'):
	# accuracy
	    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# create a summary for our cost and accuracy
	tf.summary.scalar("cost", cross_entropy)
	tf.summary.scalar("accuracy", accuracy)

	# merge all summaries into a single "operation" which we can execute in a session 
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	variables_check_op=tf.report_uninitialized_variables()
	#print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
				global_step=global_step,
				init_op=init_op)
    state = False
    #begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
	#changed by andy
	# is chief
	#if FLAGS.task_index != 0:
	    #sv.start_queue_runners(sess, [chief_queue_runner])
	    #sess.run(init_token_op)
	while(not state):
	    #print ("waiting all variables initialized!")
	    #session.run(tf.initialize_all_variables())
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
	while( final_accuracy < targted_accuracy ):
	    # number of batches in one epoch
	    batch_count = int(mnist.train.num_examples/batch_size)
	    count = 0
	    for i in range(batch_count):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
				
		# perform the operations we defined earlier on batch
		_, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step], feed_dict={x: batch_x, y_: batch_y})
		final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
		if (final_accuracy > targted_accuracy):
		    break
		#writer.add_summary(summary, step)

		count += 1
	    	if count % frequency == 0 or i+1 == batch_count:
		    elapsed_time = time.time() - start_time
		    start_time = time.time()
		    '''
		    print("Step: %d," % (step+1), 
			  " Accuracy: %.4f," % final_accuracy, 
			  " Batch: %3d of %3d," % (i+1, batch_count), 
			  " Cost: %.4f," % cost, 
			  " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
		    '''
		    count = 0
	print(str(n_PS) + '-' + str(n_Workers) + '-' + str(FLAGS.task_index) + ',' + str(step) + ',' + str(float(time.time()-begin_time)) + ',' + str(cost) + ',' + str(final_accuracy))
	
	'''
	print("sum_step: %2.2f" % step)
	print("Total Time: %3.2fs" % float(time.time() - begin_time))
	print("Final Cost: %.4f" % cost)
	print("Final accuracy: %.4f" % final_accuracy)
	'''

    sv.stop()
#    print("done")
