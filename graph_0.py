import tensorflow as tf
import numpy as np
import yaml

from helpers import *



modelName = "2xCONVTUNE_2xCONV_1xFC"



is_train = tf.placeholder_with_default(False, ())
is_tune = tf.placeholder_with_default(False, ())


global_step = tf.Variable(0, trainable=False)

learning_rate = tf.placeholder_with_default(0.01, ())
tf.summary.scalar('learning_rate', learning_rate)


features_pre_train, labels_pre_train = input_pipeline(tf.train.match_filenames_once("./data_train/*.csv"), batch_size = 150)
features_pre_test, labels_pre_test = input_pipeline(tf.train.match_filenames_once("./data_test/*.csv"), batch_size = 150)

features_pre = tf.cond(is_train, lambda: features_pre_train, lambda: features_pre_test)
labels_pre = tf.cond(is_train, lambda: labels_pre_train, lambda: labels_pre_test)

features = tf.placeholder_with_default(features_pre, shape=(None, seq_length, num_sensors))
labels = tf.placeholder_with_default(labels_pre, shape=(None, num_classes))



print(features)

###

layer, conv_b = conv(features, "conv0", 1, 1, 16, "tune_vars")
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv1", 5, 2, 16)
layer = tf.nn.relu(layer + conv_b)
print(layer)

layer, conv_b = conv(layer, "conv2", 5, 2, 16)
layer = tf.nn.relu(layer + conv_b)
print(layer)

###

layer = tf.reshape(layer, [-1, 10 * 16])
layer = tf.layers.dropout(layer, training=is_train)
print(layer)

layer, fc_b = fc(layer, "fc1", 3)
layer = layer + fc_b
print(layer)

###

probs = tf.nn.softmax(layer)
prediction_max = tf.argmax(probs, 1)
label_max = tf.argmax(labels, 1)
correct_prediction = tf.equal(prediction_max, label_max) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###

cost = tf.losses.softmax_cross_entropy(labels, layer)
tf.summary.scalar('error', cost)

summary_test = tf.summary.scalar('error_test',  tf.cond(is_train, lambda: tf.identity(0.0), lambda: tf.identity(cost)))

l2_losses = tf.reduce_sum(tf.get_collection("l2_losses")) * 0.01
tf.summary.scalar('l2_losses', l2_losses)

total_cost = cost + l2_losses

###

pretrain = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_cost, global_step=global_step)
finetune = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_cost, var_list=tf.get_collection("tune_vars"))

train = tf.cond(is_tune, lambda: finetune, lambda: pretrain)

###

summary = tf.summary.merge_all()
