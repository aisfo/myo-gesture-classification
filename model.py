import tensorflow as tf
import numpy as np
import yaml

from helpers import *



modelName = "2xCONVTUNE_2xCONV_1xFC"



is_train = tf.placeholder_with_default(False, ())
is_tune = tf.placeholder_with_default(False, ())
is_fed = tf.placeholder_with_default(False, ())

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.placeholder_with_default(0.01, ())
tf.summary.scalar('learning_rate', learning_rate)

features_pretrain, labels_pretrain = input_pipeline(tf.train.match_filenames_once("./data_train/pretrain*.csv"), batch_size = 150)
features_pretrain_test, labels_pretrain_test = input_pipeline(tf.train.match_filenames_once("./data_test/pretrain*.csv"), batch_size = 150)

features_tune, labels_tune = input_pipeline(tf.train.match_filenames_once("./data_train/tune_train*.csv"), batch_size = 75)
features_tune_test, labels_tune_test = input_pipeline(tf.train.match_filenames_once("./data_test/tune_test*.csv"), batch_size = 225)

features_feed = tf.placeholder(tf.float32, shape=(None, seq_length, num_sensors))
labels_feed = tf.placeholder(tf.float32, shape=(None, num_classes))


features_train = tf.cond(is_tune, lambda: features_tune, lambda: features_pretrain)
labels_train = tf.cond(is_tune, lambda: labels_tune, lambda: labels_pretrain)

features_test = tf.cond(is_tune, lambda: features_tune_test, lambda: features_pretrain_test)
labels_test = tf.cond(is_tune, lambda: labels_tune_test, lambda: labels_pretrain_test)

features = tf.cond(is_train, lambda: features_train, lambda: features_test)
labels = tf.cond(is_train, lambda: labels_train, lambda: labels_test)

# features = tf.cond(is_fed, lambda: features_feed, lambda: features)
# labels = tf.cond(is_fed, lambda: labels_feed, lambda: labels)



print(features)

###

layer, conv_b = conv(features, "conv_pre1", 10, 1, 8, "tune_vars")
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv_pre2", 3, 1, 8, "tune_vars")
layer = tf.nn.relu(layer + conv_b)
print(layer)

###

layer, conv_b = conv(layer, "conv1", 3, 2, 16)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv2", 3, 2, 32)
layer = tf.nn.relu(layer + conv_b)
print(layer)

###

layer = tf.reshape(layer, [-1, 10 * 32])
layer = tf.layers.dropout(layer, training=is_train)
print(layer)

layer, fc_b = fc(layer, "fc1", 3)
layer = layer + fc_b
print(layer)

###

prediction_max = tf.argmax(tf.nn.softmax(layer), 1)
label_max = tf.argmax(labels, 1)
correct_prediction = tf.equal(prediction_max, label_max) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###

cost = tf.losses.softmax_cross_entropy(labels, layer)
tf.summary.scalar('error', cost)

total_cost = cost + tf.reduce_sum(tf.get_collection("l2_losses")) * 0.001
tf.summary.scalar('error_total', total_cost)

###

pretrain = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_cost, global_step=global_step)
tune = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_cost, var_list=tf.get_collection("tune_vars"))

train = tf.cond(is_tune, lambda: tune, lambda: pretrain)

###

summary = tf.summary.merge_all()
