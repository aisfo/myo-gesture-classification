from queue import Queue
import myo
import threading
import time
import signal
import sys
import curses
import uuid

myo.init()
hub = myo.Hub()
key=""
curfile = None

class MyListener(myo.DeviceListener):
  def on_connect(self, device, timestamp, firmware_version):
    device.set_stream_emg(myo.StreamEmg.enabled)

  def on_emg_data(self, device, timestamp, emg_data):
    if curfile is not None:
      csv_line = ",".join(map(str, emg_data)) + "\n"
      curfile.write(csv_line)

listener = MyListener()



def exit(signal, frame):
      hub.shutdown()
      sys.exit(0)

signal.signal(signal.SIGINT, exit)


if not os.path.exists('data'):
    os.makedirs('data')


def main(win):
    win.nodelay(True)
    win.clear()   
    win.addstr("Press 1, 2 or 3 to start recording data for that class")

    while True: 
      try:
        key = win.getkey()
        if str(key) in ["1", "2", "3"]:
          random_id = str(uuid.uuid4()).replace("-","")[:10]
          curfile = open('data/recording_{0}_{1}.csv'.format(random_id, key),'a')
          win.clear()
          win.addstr("Recording class for <{0}>".format(key))
          hub.run(1000, listener)
        else:
          hub.stop(True)
          if curfile is not None:
            curfile.close()
            curfile = None
          win.clear()
          win.addstr("Press 1, 2 or 3 to start recording data for that class")
          
      except Exception as e:
        pass


curses.wrapper(main)