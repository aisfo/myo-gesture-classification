import tensorflow as tf
import numpy as np
import model as m
from sklearn import metrics

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())

  modelName = m.modelName
  summary_writer = tf.summary.FileWriter('summary/{0}'.format(modelName), graph=sess.graph)

  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for i in range(8 * 100):
    # Retrieve a single instance:
    _, train_cost, train_accuracy, summary, step = sess.run([m.train, m.cost, m.accuracy, m.summary, m.global_step], 
        feed_dict={ m.is_train: True, m.is_tune: False })
    summary_writer.add_summary(summary, step)
    
    if i % (8 * 10) == 0:
      test_cost, test_accuracy, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
          feed_dict={ m.is_train: False, m.is_tune: False })
      print("{0} :: train( {1} {2} ) test( {3} {4} )".format(i, train_cost, train_accuracy, test_cost, test_accuracy))
      print(metrics.confusion_matrix(label, prediction))

      test_cost, test_accuracy, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
          feed_dict={ m.is_train: False, m.is_tune: True })
      print(":::: test_tune( {3} {4} )".format(i, train_cost, train_accuracy, test_cost, test_accuracy))
      print(metrics.confusion_matrix(label, prediction))

  for i in range(1):
    # Retrieve a single instance:
    _, train_cost, train_accuracy, summary, step = sess.run([m.train, m.cost, m.accuracy, m.summary, m.global_step], 
        feed_dict={ m.is_train: True, m.is_tune: True })
    summary_writer.add_summary(summary, step)
    
    test_cost, test_accuracy, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
        feed_dict={ m.is_train: False, m.is_tune: False })
    print("tune :: train_tune( {1} {2} ) test( {3} {4} )".format(i, train_cost, train_accuracy, test_cost, test_accuracy))
    print(metrics.confusion_matrix(label, prediction))

    test_cost, test_accuracy, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
        feed_dict={ m.is_train: False, m.is_tune: True })
    print(":::: test_tune( {3} {4} )".format(i, train_cost, train_accuracy, test_cost, test_accuracy))
    print(metrics.confusion_matrix(label, prediction))

  coord.request_stop()
  coord.join(threads)