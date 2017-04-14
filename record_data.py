from queue import Queue
import myo
import threading
import time
import signal
import sys
import curses
import uuid
import os
import yaml
import argparse


config = yaml.safe_load(open("config.yml"))
seq_length = config["seq_length"]
seq_per_file = 50


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=False)
args = parser.parse_args()


key = ""
curfile = None
defaultText = "Press 1(rock), 2(paper) or 3(scissors) to start recording data for that class"


myo.init()
hub = myo.Hub()

  
def exit(signal, frame):
  hub.shutdown()
  sys.exit(0)

signal.signal(signal.SIGINT, exit)

class MyListener(myo.DeviceListener):

  def __init__(self, win):
    self.file_size = 0
    self.win = win

  def on_connect(self, device, timestamp, firmware_version):
    device.set_stream_emg(myo.StreamEmg.enabled)

  def on_emg_data(self, device, timestamp, emg_data):
    global curfile

    if curfile is not None and self.file_size >= seq_length * seq_per_file:
      curfile.close()
      curfile = None
      self.file_size = 0

      self.win.clear()
      self.win.addstr(defaultText)

    if curfile is not None:
      csv_line = ",".join(map(str, emg_data))
      csv_line = "{0},{1}\n".format(csv_line, key)
      curfile.write(csv_line)
      self.file_size += 1



def main(win):
  global key
  global curfile

  win.nodelay(True)
  win.clear()
  win.addstr(defaultText)

  listener = MyListener(win)
  hub.run(200, listener)

  while True: 
    try:
      key = win.getkey()

      if str(key) in ["1", "2", "3"]:
        win.clear()
        win.addstr("Recording class for <{0}>".format(key))

        random_id = str(uuid.uuid4()).replace("-","")[:10]
        curfile = open('data_train/{0}_{1}_{2}.csv'.format(key, args.prefix, random_id), 'a')
        
    except Exception as e:
      pass


curses.wrapper(main)