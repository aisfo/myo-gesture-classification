import time
import signal
import sys
import uuid
import os
import argparse

import tensorflow as tf
import numpy as np
import myo
import curses
import yaml

import model as m


parser = argparse.ArgumentParser()
parser.add_argument("--load", required=True)
args = parser.parse_args()



config = yaml.safe_load(open("config.yml"))

SEQUENCE_LEN = config["seq_length"]

CLASS_KEYS = ["1", "2", "3"]
TUNE_KEY = "t"
RECORD_KEY = "r"






key=""

tune_text = "Press 1(rock), 2(paper) or 3(scissors) to record data for that class"



saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  
  saver.restore(sess, "./models/" + args.load)

  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  myo.init()
  hub = myo.Hub()
    

  def exit(signal, frame):
    print("exiting")
    hub.stop(True)
    hub.shutdown()

    coord.request_stop()
    coord.join(threads)

    sys.exit(0)

  signal.signal(signal.SIGINT, exit)


  def tune_model(tune_data):
    features = []
    labels = []

    for class_data, _class in enumerate(tune_data):
      for class_data_point in class_data:
        one_hot_class = [ 0, 0, 0 ]
        one_hot_class[_class] = 1
        labels.append(one_hot_class)
        features.append(class_data_point)

    for i in range(101):
      _, train_cost, train_accuracy = sess.run([m.train, m.cost, m.accuracy], 
        feed_dict={ 
          m.is_train: True, 
          m.is_tune: True, 
          m.features: features, 
          m.labels: labels,
          m.learning_rate: 0.001 
        })


  class MyListener(myo.DeviceListener):

      def __init__(self, win):
        self.buffer = []
        self.win = win
        self.tune_data = [[], [], []]
        self.state = "predicting"


      def get_data(self):
        return self.tune_data


      def set_state(self, _state):
        self.state = _state


      def get_state(self):
        return self.state


      def clear_buffer():
        self.buffer = []


      def clear_data(_class):
        self.tune_data[_class] = []


      def on_connect(self, device, timestamp, firmware_version):
        device.set_stream_emg(myo.StreamEmg.enabled)


      def on_emg_data(self, device, timestamp, emg_data):
        if self.state is "recording" and key in CLASS_KEYS:
          _class = int(key) - 1

          self.buffer.append(emg_data)
          if len(self.buffer) < SEQUENCE_LEN: return
          
          self.tune_data[_class].append(self.buffer)
          if len(self.tune_data[_class]) == 25:
            self.set_state("waiting")
            self.win.clear()
            self.win.addstr(tune_text)
            return

          self.clear_buffer()

        if self.state is "predicting":
          self.buffer.append(emg_data)
          if len(self.buffer) < SEQUENCE_LEN: return

          prediction = sess.run(m.probs, feed_dict={ m.features: [ self.buffer ] })

          self.clear_buffer()
          
          probabilities = prediction[0][0]
          self.win.clear()
          self.win.addstr("classification: {0} => {1}".format([format(a, '.4f') for a in probabilities], np.argmax(probabilities)))


  def main(win):
    global key

    win.nodelay(True)
    win.clear()

    listener = MyListener(win)
    hub.run(200, listener)

    while True: 
      try:
        state = listener.get_state()
        if state is "recording": continue

        key = str(win.getkey())

        if state is "predicting" and key == RECORD_KEY:
          listener.set_state("waiting")
          listener.clear_buffer()
          win.clear()
          win.addstr(tune_text)

        elif state is "waiting" and key in CLASS_KEYS:
          listener.set_state("recording")
          win.clear()
          win.addstr("Recording data for class <{0}>...".format(key))

        elif state is "waiting" and key == TUNE_KEY:
          listener.set_state("tuning")
          listener.clear_buffer()
          win.clear()
          win.addstr("Tuning...")

          tune_data = listener.get_data()
          tune_model(tune_data)
          listener.clear_data()

          listener.set_state("predicting")

      except curses.error as e:
        pass


  curses.wrapper(main)

  