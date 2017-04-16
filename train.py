import tensorflow as tf
import numpy as np
import model as m
import argparse
import time
from sklearn import metrics



parser = argparse.ArgumentParser()
parser.add_argument("--load", required=False)
parser.add_argument("--save", required=False)
args = parser.parse_args()


saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

  if args.load:
    saver.restore(sess, "./models/" + args.load)
  else:
    sess.run(tf.global_variables_initializer())

  modelName = "{0}_{1}".format(m.modelName, round(time.time()))
  summary_writer = tf.summary.FileWriter('summary/{0}'.format(modelName), graph=sess.graph)
  summary_writer_test = tf.summary.FileWriter('summary/{0}_test'.format(modelName), graph=sess.graph)

  # start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)


  # PRETRAINING
  for i in range(10001):

    _, train_cost, train_accuracy, summary, step = sess.run([m.train, m.cost, m.accuracy, m.summary, m.global_step], 
        feed_dict={ m.is_train: True, m.is_tune: False, m.learning_rate: 0.01 })

    summary_writer.add_summary(summary, step)
    
    if i % 100 == 0:
      test_cost, test_accuracy, summary = sess.run([m.cost, m.accuracy, m.summary_test], 
          feed_dict={ m.is_train: False, m.is_tune: False })

      summary_writer_test.add_summary(summary, step)

      print("TRAIN :: pre_train {1:3.4f} {2:3.4f} :: pre_test {3:3.4f} {4:3.4f} :: {0}"
        .format(i, train_cost, train_accuracy, test_cost, test_accuracy)) 

      if args.save:
        save_path = saver.save(sess, "./models/" + args.save)
  

  # stop input queue
  coord.request_stop()
  coord.join(threads)