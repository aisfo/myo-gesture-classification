import tensorflow as tf
import numpy as np


num_sensors = 8
seq_length = 40
num_classes = 3
modelName = "conv_1"



def input_pipeline(filenames, batch_size):
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)

  reader = tf.TextLineReader()
  keys, values = reader.read_up_to(filename_queue, seq_length)

  record_defaults = [[0]] * (num_sensors + 1)
  csv_line = tf.decode_csv(values, record_defaults=record_defaults)

  features = tf.stack(csv_line[:num_sensors], axis=1)
  features = tf.to_float(features)
  mean, var = tf.nn.moments(features, axes=[0, 1])
  features = (features - mean) / (tf.sqrt(var) + 1e-5)

  label = csv_line[-1][0] - 1
  label = tf.one_hot(label, depth=num_classes)

  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  feature_batch, label_batch = tf.train.shuffle_batch([features, label], 
      batch_size=batch_size, 
      shapes=[[seq_length, num_sensors], [num_classes]],
      capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  return feature_batch, label_batch


def conv(_input, name, width, stride, out_depth):
  with tf.variable_scope(name):
    tf.summary.histogram("in", _input)

    input_shape = _input.get_shape().as_list()
    in_depth = input_shape[-1]

    conv_shape = [width, in_depth, out_depth]

    stddev = np.sqrt(2.0 / (width * out_depth))
    conv_w = tf.get_variable("w", conv_shape, initializer=tf.random_normal_initializer(stddev=stddev))
    conv_b = tf.get_variable("b", out_depth, initializer=tf.constant_initializer(0))
     
    tf.add_to_collection("l2_losses", tf.nn.l2_loss(conv_w))

    _input = tf.nn.conv1d(_input, conv_w, stride, padding='SAME')

    tf.summary.histogram("out", _input)
    tf.summary.histogram("w", conv_w)
    tf.summary.histogram("b", conv_b)

    return _input, conv_b


def fc(_input, name, out_depth):
  with tf.variable_scope(name):
    tf.summary.histogram("in", _input)

    input_shape = _input.get_shape().as_list()
    in_depth = input_shape[-1]

    fc_w = tf.get_variable("fc1_w", [in_depth, out_depth], initializer=tf.contrib.layers.xavier_initializer())
    fc_b = tf.get_variable("fc1_b", [out_depth], initializer=tf.constant_initializer(0))

    tf.add_to_collection("l2_losses", tf.nn.l2_loss(fc_w))

    _input = tf.matmul(_input, fc_w)

    tf.summary.histogram("out", _input)
    tf.summary.histogram("w", fc_w)
    tf.summary.histogram("b", fc_b)

    return _input, fc_b



is_train = tf.placeholder_with_default(True, ())
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 1.0, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

features, labels = input_pipeline(tf.train.match_filenames_once("./data/*.csv"), batch_size = 150)
features_test, labels_test = input_pipeline(tf.train.match_filenames_once("./test/*.csv"), batch_size = 150)

signal = tf.cond(is_train, lambda: features, lambda: features_test)
labels = tf.cond(is_train, lambda: labels, lambda: labels_test)


layer, conv_b = conv(signal, "conv1", 3, 1, 32)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv2", 3, 1, 32)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer = tf.layers.max_pooling1d(layer, 5, 2, padding='same', name="pool1")

layer, conv_b = conv(layer, "conv3", 3, 1, 64)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv4", 3, 1, 64)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer = tf.layers.max_pooling1d(layer, 5, 2, padding='same', name="pool2")

layer, conv_b = conv(layer, "conv5", 3, 1, 128)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv6", 3, 1, 128)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer = tf.reshape(layer, [-1, 10 * 128])
layer = tf.layers.dropout(layer, training=is_train)
print(layer)

layer, fc_b = fc(layer, "fc1", 10 * 32)
layer = tf.nn.relu(layer + fc_b)
print(layer)

layer = tf.layers.dropout(layer, training=is_train)
layer, fc_b = fc(layer, "fc2", 3)
layer = layer + fc_b
print(layer)

prediction_max = tf.argmax(tf.nn.softmax(layer), 1)
label_max = tf.argmax(labels, 1)
correct_prediction = tf.equal(prediction_max, label_max) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost = tf.losses.softmax_cross_entropy(labels, layer)
tf.summary.scalar('error', cost)
total_cost = cost + tf.reduce_sum(tf.get_collection("l2_losses")) * 0.001
tf.summary.scalar('error_total', total_cost)

train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_cost, global_step=global_step)

summary = tf.summary.merge_all()
