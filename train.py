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

  for i in range(1001):
    # Retrieve a single instance:
    _, train_cost, train_accuracy, summary, step = sess.run([m.train, m.cost, m.accuracy, m.summary, m.global_step], 
        feed_dict={ m.is_train: True, m.is_tune: False, m.learning_rate: 0.01 })
    summary_writer.add_summary(summary, step)
    
    if i % 50 == 0:
      test_cost, test_accuracy = sess.run([m.cost, m.accuracy], 
          feed_dict={ m.is_train: False, m.is_tune: False })
      
      test_cost_tune, test_accuracy_tune = sess.run([m.cost, m.accuracy], 
          feed_dict={ m.is_train: False, m.is_tune: True })

      print("train {1:3.4f} {2:3.4f} :: test {3:3.4f} {4:3.4f} :: tune {5:3.4f} {6:3.4f} :: {0}"
        .format(i, train_cost, train_accuracy, test_cost, test_accuracy, test_cost_tune, test_accuracy_tune)) 


  print("\n---------------------------------\n")

  i = 0
  while (False):
    # Retrieve a single instance:
    _, train_cost_tune, train_accuracy_tune = sess.run([m.train, m.cost, m.accuracy], 
        feed_dict={ m.is_train: True, m.is_tune: True, m.learning_rate: 0.001 })
    #summary_writer.add_summary(summary, step)

    _, train_cost, train_accuracy = sess.run([m.train, m.cost, m.accuracy], 
        feed_dict={ m.is_train: True, m.is_tune: False, m.learning_rate: 0.001 })
    # summary_writer.add_summary(summary, step)
    
    if i % 10 == 0:
      test_cost, test_accuracy, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
          feed_dict={ m.is_train: False, m.is_tune: False })
     

      test_cost_tune, test_accuracy_tune, prediction, label = sess.run([m.cost, m.accuracy, m.prediction_max, m.label_max], 
          feed_dict={ m.is_train: False, m.is_tune: True })
      
      print("train {1:3.4f} {2:3.4f} :: test {3:3.4f} {4:3.4f} :: tune {5:3.4f} {6:3.4f} :: {0}"
        .format(i, train_cost_tune, train_accuracy_tune, test_cost, test_accuracy, test_cost_tune, test_accuracy_tune, i))
      #print(metrics.confusion_matrix(label, prediction))

    i += 1

    if train_accuracy_tune > 0.98:
      break

  coord.request_stop()
  coord.join(threads)