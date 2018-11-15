# LIVE_SER
Live demo for speech emotion recognition and laughter detection using Keras and Tensorflow models.

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

<a id="top"/>

This folder has source codes for speech emotion recognition and laughter detection. This module relies on many machine learning packages and platforms such as Google Tensorflow and Keras, which is computationally expensive. Hence, it may not be operationable on mobile devices. Performance depends on contextual factors such as speaker, language, environment, etc

The module mainly consists of two parts: voice activity detection and recognition (see details in codes). At this moment, we provide an emotion recognition model that was trained on aggregated English speech corpora (eNTERFACE, SEMAINE, and IEMOCAP), so that English speaker may best fit this model. However, you can use your own trained model for the best performance.

The prediction is very sensitive to gains of the microphone. Hence, it is important to set minimum and maximum gains for the normalisation of gains. Please find details of arguments below.

## Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--build">Build</a>

3. <a href="#3--device">Device setup</a>

4. <a href="#4--usage">Usage</a>

5. <a href="#5--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>
This software only runs on OSX or Linux (tested on Ubuntu). It is compatible with python 2.x and 3.x, but the following descriptions assume that python 3.x is installed.

### basic system packages

This software relies on several system packages that must be installed using a software manager.

For Ubuntu, please run the following steps:

```bash
sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev
```

For mac osx, you should install portaudio and pulseaudio using brew as follows:
(if you do not have brew, then type:
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)")

brew install portaudio

brew install pulseaudio

If you have any issues with portaudio, see:
https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

### virtual environment
To avoid conflicts of softwares, we recommend you to use virtualenv (see:
https://virtualenv.pypa.io/en/stable/).

### python packages
Using pip, install all pre-required modules.
(pip version >= 8.1 is required, see: http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest)

If you have old numpy (<1.12) please remove it.
https://github.com/tensorflow/tensorflow/issues/559

Next, install a right version of tensorflow depending on your os (see https://www.tensorflow.org/install/)

For example, you can install a cpu version for mac os:

pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl 

For python2,
pip2 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py2-none-any.whl 

If you have older osx ( < 10.12.6), you should install older versions of tensorflow like:

pip install tensorflow==1.5

Then, to install other required packages,

sudo pip install -r requirements.txt

## 2. Build <a id="2--build"/>
Any building process is not required at the moment.

## 3. Device setup <a id="3--device"/>

### Running on Linux
Currently, using pulseaudio as the input device is the most stable way. If you do not specify device ID, a pulse audio device will be chosen as a default input. However, you must make sure if pulseaudio server is running. (if not, type "pulseaudio --start").

If you want and know pulse & alsa works, you can choose your own input device as a pulse audio and use the pulse as the input device for emotion recognition as follows:

0. run pulseaudio as a daemon process if it's off

```bash
pulseaudio --D
```

1. find your device (see: https://wiki.archlinux.org/index.php/PulseAudio/Examples) by typing in your terminal.

```bash
pacmd list-sources | grep -e device.string -e 'name:'
```

Depending on os and devices, it gives you various names. You need to choose a right input device among them.

For example, you find "name: <Channel_1__Channel_2.4>" as your input source.
Then, set a default device for "pulse" by typing:

```bash
pacmd "set-default-source Channel_1__Channel_2.4"
```

2. run 

The following script will use the default pulse device

```bash
python ./src/offline_ser.py -p_mode 2 -f_mode 1 -log ./output/live.csv \
       -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5            \
       -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'
```

### Running on OSX

Type:

python3 ./src/offline_ser.py

This will give you a list of audio devices and you need to identify index of your device.
Note that this index changes depending on usb devices being used. Hence, it's safe to check before it runs.

if you find your device in index 2 (for the argument of d_id), run:

python ./src/offline_ser.py -d_id 2 -p_mode 2 -f_mode 1 -log ./output/live.csv -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'


## 4. Usage <a id="4--usage"/>

### LIVE MODE
There are many parameters that controls VAD, feature extraction, and prediction. To get information of parameters, type:

```bash
python ./src/offline_ser.py 
```

For quick use (assuming your device id is 2):

```bash
python ./src/offline_ser.py -d_id 2 -p_mode 2 -f_mode 1 -log ./output/live.csv \
       -md ./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 \
       -tasks 'arousal:3,valence:3' --seq2seq
```

The provided model (./model/si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5) is trained by using end-to-end method, which means its input feature is raw-wave. So you have to specify -f_mode as "1". Also, the raw wave form has 16000 samples per sec. So we set -m_t_step as "16000". The model uses 10 contextual windows; so each window has 1600 samples (-c_len 1600).

To get the probablistic distribution, we set -p_mode as "2".

The model predicts two tasks: arousal and valence. Each task has 3 classes (low, neutral, and high). So we specify -tasks as "arousal:3,valence:3".

Note that gain normalisation is crucial to the performance. Details can be found in ./scripts/decode_file.sh

### BATCH MODE
You can also put a wave file instead of using a live microphone:

```bash
python ./src/offline_ser.py -d_id 1 -vd 1000 -p_mode 2 -f_mode 1 -log ./output/offline.wav.csv \
       -md ../model/ami.raw.cnnlstmfcn.0.h5 -c_len 1600 -m_t_step 16000 -tasks 'laughter:2' --stl \
       --save -wav './wav/your_wave.wav'
```

If you run a batch mode, putting a list of wave files, make a txt file that contains a list of wave files, first. Wave files are separated by newlines.

```bash
python ./src/offline_ser.py -d_id 1 -vd 1000 -p_mode 2 -f_mode 1 -log ./output/offline.wav.csv \
       -md ../model/ami.raw.cnnlstmfcn.0.h5 -c_len 1600 -m_t_step 16000 -tasks 'laughter:2' --stl \
       --save -batch './wav/list_wave.csv'
```

## 5. References <a id="5--references"/>

This software is based on the following papers. Please cite one of these papers in your publications if it helps your research:

@inproceedings{kim2017interspeech,
  title={Towards Speech Emotion Recognition ``in the wild'' using Aggregated Corpora and Deep Multi-Task Learning},
  author={\textbf{Kim, Jaebok} and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa},
  booktitle={Proceedings of the INTERSPEECH},
  pages={1113--1117},
  year={2017}
}


@inproceedings{kim2017acmmm, title={Deep Temporal Models using Identity Skip-Connections for Speech Emotion Recognition}, author={Kim, Jaebok and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa}, booktitle={Proceedings of ACM Multimedia}, pages={1006-1013}, year={2017} }

@inproceedings{kim2017acii, title={Learning spectro-temporal features with 3D CNNs for speech emotion recognition}, author={Kim, Jaebok and Truong, Khiet and Englebienne, Gwenn and Evers, Vanessa}, booktitle={Proceedings of International Conference on Affective Computing and Intelligent Interaction}, pages={}, year={2017} }

<a id="top"/> 
