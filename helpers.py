import tensorflow as tf
import numpy as np
import yaml


config = yaml.safe_load(open("config.yml"))

NUM_SENSORS = config["num_sensors"]
SEQUENCE_LEN = config["sequence_len"]
NUM_CLASSES = config["num_classes"]



def input_pipeline(filenames, batch_size):
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

  reader = tf.TextLineReader()
  keys, values = reader.read_up_to(filename_queue, SEQUENCE_LEN)

  record_defaults = [[0]] * (NUM_SENSORS + 1)
  csv_line = tf.decode_csv(values, record_defaults=record_defaults)

  features = tf.stack(csv_line[:NUM_SENSORS], axis=1)
  features = tf.to_float(features)
  mean, var = tf.nn.moments(features, axes=[0, 1])
  features = (features - mean) / (tf.sqrt(var) + 1e-5)

  label = csv_line[-1][0] - 1
  label = tf.one_hot(label, depth=NUM_CLASSES)

  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  feature_batch, label_batch = tf.train.shuffle_batch(
      [features, label], 
      batch_size=batch_size, 
      shapes=[[SEQUENCE_LEN, NUM_SENSORS], [NUM_CLASSES]],
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  return feature_batch, label_batch



def conv(_input, name, width, stride, out_depth, collection=None):
  with tf.variable_scope(name):
    tf.summary.histogram("in", _input)

    input_shape = _input.get_shape().as_list()
    in_depth = input_shape[-1]

    conv_shape = [width, in_depth, out_depth]

    stddev = np.sqrt(2.0 / (width * out_depth))
    conv_w = tf.get_variable("w", conv_shape, initializer=tf.random_normal_initializer(stddev=stddev))
    conv_b = tf.get_variable("b", out_depth, initializer=tf.constant_initializer(0))
     
    tf.add_to_collection("l2_losses", tf.nn.l2_loss(conv_w))

    if collection is not None:
      tf.add_to_collection(collection, conv_w)
      tf.add_to_collection(collection, conv_b)

    _output = tf.nn.conv1d(_input, conv_w, stride, padding='SAME')

    tf.summary.histogram("out", _output)
    tf.summary.histogram("w", conv_w)
    tf.summary.histogram("b", conv_b)

    return _output, conv_b



def fc(_input, name, out_depth):
  with tf.variable_scope(name):
    tf.summary.histogram("in", _input)

    input_shape = _input.get_shape().as_list()
    in_depth = input_shape[-1]

    fc_w = tf.get_variable("w", [in_depth, out_depth], initializer=tf.contrib.layers.xavier_initializer())
    fc_b = tf.get_variable("b", [out_depth], initializer=tf.constant_initializer(0))

    tf.add_to_collection("l2_losses", tf.nn.l2_loss(fc_w))

    _output = tf.matmul(_input, fc_w)

    tf.summary.histogram("out", _output)
    tf.summary.histogram("w", fc_w)
    tf.summary.histogram("b", fc_b)

    return _output, fc_b



