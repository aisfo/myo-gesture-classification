import time
import signal
import sys
import uuid
import os
import argparse

import myo
import curses
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--prefix", required=True)
args = parser.parse_args()


config = yaml.safe_load(open("config.yml"))

SEQUENCE_LEN = config["sequence_len"]
SEQUENCE_PER_FILE = 200 * 10 // SEQUENCE_LEN


os.makedirs("data_train", exist_ok=True)


key = ""
curfile = None
defaultText = "Press 1(rock), 2(paper) or 3(scissors) to start recording data for that class"


myo.init()
hub = myo.Hub()

def exit(signal, frame):
  print("exiting")
  hub.stop(True)
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

    if curfile is not None and self.file_size >= SEQUENCE_LEN * SEQUENCE_PER_FILE:
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

      if curfile is not None: 
        continue

      if str(key) in ["1", "2", "3"]:
        win.clear()
        win.addstr("Recording data for <{0}>".format(key))

        random_id = str(uuid.uuid4()).replace("-","")[:10]
        curfile = open('data_train/{0}_{1}_{2}.csv'.format(args.prefix, key, random_id), 'a')
        
    except Exception as e:
      pass


curses.wrapper(main)