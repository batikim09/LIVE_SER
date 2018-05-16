# LIVE_SER
Live demo for speech emotion recognition using Keras and Tensorflow models

<a id="top"/>

This folder has source codes for speech emotion recognition. This module relies on many machine learning packages and platforms such as Google Tensorflow and Keras, which is comptutationally expensive. Hence, it may not be operationable on mobile devices. Performance depends on contextual factors such as speaker, language, environment, etc. 

The module mainly consists of two parts: voice activity detection and recognition (see details in codes). At this moment, we provide an emotion recognition model that was trained on aggregated English speech corpora (LDC prosody, eNTERFACE, SEMAINE, and IEMOCAP), so that English speaker may fit this model. However, you can use your own trained model.

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

##Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--build">Build</a>

3. <a href="#3--device">Device setup</a>

4. <a href="#4--usage">Usage</a>

5. <a href="#5--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>

This software is compatible with only python 2.x and 3.x, but the following descrptions assume 3.x.

### basic system packages
Please run the following steps:

`sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev' for Ubuntu

For mac osx, you should install portaudio and pulseaudio using brew.

brew install portaudio

brew install pulseaudio

If you have any issues with portaudio, see:
https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

### python packages
Using pip, install all pre-required modules.
(pip version >= 8.1 is required, see: http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest)

If you have old numpy (<1.12) please remove it.
https://github.com/tensorflow/tensorflow/issues/559

Next, install a right version of tensorflow depending on your os (see https://www.tensorflow.org/install/)

For example, you can install a cpu version for mac os:

pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl 
 
Then,

sudo pip3 install -r requirements.txt

If you have numpy already, it must be higher than 1.12.0

## 2. Build <a id="2--build"/>
Any building processing is not required at the moment.

## 3. Device setup <a id="3--device"/>
Currently, using pulse audio as the input device is the best stable way. If you do not specify device ID, a pulse audio device will be chosen as an input. However, you must make sure if pulseaudio server is running. (if not, type "pulseaudio --start").

If you want and know pulse & alsa works, you can choose your own input device as a pulse audio and use the pulse as the input device for emotion recognition as follows:

0. run pulseaudio as a daemon process if it's off

pulseaudio --D

1. find your device (see: https://wiki.archlinux.org/index.php/PulseAudio/Examples) by 

pacmd list-sources | grep -e device.string -e 'name:'

Depending on os and devices, it gives you various names. You need to choose a right input device among them.

2. set a default device for "pulse" by typing in a terminal for example:

pacmd "set-default-source alsa_input.usb-Andrea_Electronics_Corp._Andrea_Stereo_USB_Mic-00-Mic.analog-stereo"

3. check it works:

pacmd stat

4. check your "pulse" device's ID in pulseaudio:

rosrun squirrel_ser ser.py

This will give you a list of audio devices and you need to identify index of "pulse".
Note that this index changes depending on usb devices being used. Hence, it's safe to check before it runs.

5. set the ID of "pulse" in the launch file: ser.launch
by the argument: -d_id 

## 4. Usage <a id="4--usage"/>

For a quick start, run in the terminal:

python ./src/offline_ser.py -p_mode 2 -f_mode 1 -log ./output/live.csv -md ./model/AIBO.si.ENG.cw.raw.2d.res.lstm.gpool.dnn.1.h5 -c_len 1600 -m_t_step 16000 -tasks 'arousal:3,valence:3'


To get information of parameters, 

python ./src/offline_ser.py 

usage: offline_ser.py [-h] [-d_id DEVICE_ID] [-sr SAMPLE_RATE]
                      [-fd FRAME_DURATION] [-vm VAD_MODE] [-vd VAD_DURATION]
                      [-me MIN_ENERGY] [-wav WAVE] [-g_min G_MIN]
                      [-g_max G_MAX] [-s_ratio SPEECH_RATIO] [-fp FEAT_PATH]
                      [-md MODEL_FILE] [-elm_md ELM_MODEL_FILE]
                      [-c_len CONTEXT_LEN] [-m_t_step MAX_TIME_STEPS]
                      [-log LOG_FILE] [-tasks TASKS] [-p_mode PREDICT_MODE]
                      [-f_mode FEAT_MODE] [-f_dim FEAT_DIM] [--stl] [--save]
                      [--play]

optional arguments:
  -h, --help            show this help message and exit
  
  -d_id DEVICE_ID, --device_id DEVICE_ID
                        device id for microphone
  
  -sr SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        number of samples per sec, only accept
                        [8000|16000|32000]
  
  -fd FRAME_DURATION, --frame_duration FRAME_DURATION
                        a duration of a frame msec, only accept [10|20|30]
  
  -vm VAD_MODE, --vad_mode VAD_MODE
                        vad mode, only accept [0|1|2|3], 0 more quiet 3 more
                        noisy
  
  -vd VAD_DURATION, --vad_duration VAD_DURATION
                        minimum length(ms) of speech for emotion detection
  
  -me MIN_ENERGY, --min_energy MIN_ENERGY
                        minimum energy of speech for emotion detection
  
  -wav WAVE, --wave WAVE
                        wave file (offline mode)
  
  -g_min G_MIN, --gain_min G_MIN
                        min value of automatic gain normalisation
  
  -g_max G_MAX, --gain_max G_MAX
                        max value of automatic gain normalisation
  
  -s_ratio SPEECH_RATIO, --speech_ratio SPEECH_RATIO
                        speech ratio
  
  -fp FEAT_PATH, --feat_path FEAT_PATH
                        temporay feat path
  
  -md MODEL_FILE, --model_file MODEL_FILE
                        keras model path
  
  -elm_md ELM_MODEL_FILE, --elm_model_file ELM_MODEL_FILE
                        elm model_file
  
  -c_len CONTEXT_LEN, --context_len CONTEXT_LEN
                        context window's length
  
  -m_t_step MAX_TIME_STEPS, --max_time_steps MAX_TIME_STEPS
                        maximum time steps for DNN
  
  -log LOG_FILE, --log_file LOG_FILE
                        log
  
  -tasks TASKS, --tasks TASKS
                        tasks (arousal:2,valence:2)
  
  -p_mode PREDICT_MODE, --predict PREDICT_MODE
                        0 = diff, 1 = classification, 2 = distribution
  
  -f_mode FEAT_MODE, --feat_mode FEAT_MODE
                        0 = lspec, 1 = raw wav
  
  -f_dim FEAT_DIM, --feat_dim FEAT_DIM
                        feature dimension (# spec for lspec or mspec
  
  --stl                 only for single task learning model
  
  --save                save voice files
  
  --play                real time play


## 5. References <a id="5--references"/>


Please cite one of these papers in your publications if it helps your research:

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
