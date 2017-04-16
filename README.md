# Myo Gesture Classification 


TensorFlow models for EMG data classification. Data is recorded using The Myo Armband by Thalmic Labs.
The models are trained to recognize classes corresponding to "rock", "scissors" and "paper" gestures.



## Setup

* The scripts are tested on Mac only.
* Follow [these instructions](http://myo-python.readthedocs.io/en/latest/#installation) to install *myo-python* package.
* Connect and lock Myo armband.

## Scripts

* **record_data.py --prefix FILENAME_PREFIX** records 10 seconds long sequences for each class, saves each sequence into a separate CSV file in *data_train* directory.
* **train.py --graph GRAPH_NAME [--load MODEL_NAME] [--save MODEL_NAME]** trains the model defined in **GRAPH_NAME.py** based on data in *data_train* directory and uses the sequences from *data_test* to evaluate the model. Optional *load* and *save* flags load and save the model in *models* directory. Monitor training process by running **tensorboard** with logs from *summary* directory. 
* **demo.py --graph GRAPH_NAME --model MODEL_NAME** loads a pretrained model for **GRAPH_NAME.py** from *models* directory and starts classifying live data. Press "r" key to record short sequences of data for each class, and then "t" to finetune the model for current position of the armband.