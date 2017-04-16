import tensorflow as tf
import numpy as np
import model as m
from queue import Queue
import myo
import threading
import time
import signal
import sys
import curses
import uuid
import os
from sklearn import metrics
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--load", required=True)
args = parser.parse_args()




state = "predicting"
key=""
tune_data = {
  "1": [],
  "2": [],
  "3": []  
}


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


  class MyListener(myo.DeviceListener):

      def __init__(self, win):
        self.inference = []
        self.buffer = []
        self.win = win

      def on_connect(self, device, timestamp, firmware_version):
        device.set_stream_emg(myo.StreamEmg.enabled)

      def on_emg_data(self, device, timestamp, emg_data):
        global tune_data
        global state

        if state is "recording" and str(key) in ["1", "2", "3"]:
          _class = str(key)

          if len(self.buffer) < 40:
            self.buffer.append(emg_data)
            return
          
          tune_data[_class].append(self.buffer)
          if len(tune_data[_class]) == 25:
            state = "waiting"
            self.win.clear()
            self.win.addstr(tune_text)
            return

          self.buffer = [ emg_data ]


        if state is "predicting":
          if len(self.inference) < 40:
            self.inference.append(emg_data)
          else:
            prediction = sess.run([m.probs], feed_dict={ 
              m.is_train: False,
              m.is_tune: False,
              m.features: [self.inference]
            })
            self.inference = [ emg_data ] 
            self.win.clear()

            probabilities = prediction[0][0]
            self.win.addstr("prediction: {0} => {1}".format([format(a, '.4f') for a in probabilities], np.argmax(probabilities)))


  def main(win):
    global key
    global tune_data
    global state

    win.nodelay(True)
    win.clear()

    listener = MyListener(win)
    hub.run(200, listener)

    while True: 
      try:
        key = win.getkey()

        if state is "predicting" and str(key) == 'r':
          state = "waiting"
          win.clear()
          win.addstr(tune_text)

        if state is "waiting" and str(key) in ["1", "2", "3"]:
          state = "recording"
          win.clear()
          win.addstr("Recording data for <{0}>".format(key))

        if state is not "tuning" and str(key) == 't':
          state = "tuning"
          win.clear()
          win.addstr("Tuning...")

          features = []
          labels = []
          for _class in ["1", "2", "3"]:
            class_data = tune_data[_class]
            for class_data_point in class_data:
              one_hot_class = [0, 0, 0]
              one_hot_class[int(_class) - 1] = 1
              labels.append(one_hot_class)
              features.append(class_data_point)

          for i in range(101):
            _, train_cost, train_accuracy = sess.run([m.train, m.cost, m.accuracy], 
                feed_dict={ 
                  m.is_train: True, 
                  m.is_tune: True, 
                  m.features: features, 
                  m.labels: labels,
                  m.learning_rate: 0.001 })

          state = "predicting"

    
      except curses.error as e:
        pass


  curses.wrapper(main)

  