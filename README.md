# LIVE_SER
Live demo for speech emotion recognition using Keras and Tensorflow models

<a id="top"/> 
# squirrel_ser

This folder has source codes for deep temporal architecture-based speech emotion recognition. Note that this module relies on many machine learning packages and platforms such as Google Tensorflow and Keras, which is comptutationally expensive without GPU supports. Hence, it may not be operationable on the robot, rather deployment on an external machine is recommended. Performance varies on speakers and environment.

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

##Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--build">Build</a>

3. <a href="#3--device">Device setup</a>

4. <a href="#4--usage">Usage</a>

4. <a href="#5--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>
####Debian packages

Please run the following steps BEFORE you run catkin_make.

`sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev'

Next, using pip, install all pre-required modules.
(pip version >= 8.1 is required.)

http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest

If you have old numpy (<1.12) please remove it.
https://github.com/tensorflow/tensorflow/issues/559

Then,
sudo pip install -r requirements.txt

If you have numpy already, it must be higher than 1.12.0
try to install tensorflow using pip, then it will install the proper version of numpy.

## 2. Build <a id="2--build"/>

Please use catkin_make to build this.

## 3. Device setup <a id="3--device"/>
Currently, using pulse audio as the input device is the best stable way. If you do not specify device ID, a pulse audio device will be chosen as an input. However, you must make sure if pulseaudio server is running. (if not, type "pulseaudio --start").

If you want and know pulse & alsa works, you can choose your own input device as a pulse audio and use the pulse as the input device for emotion recognition as follows:

0. turn on pulseaudio server if it's off

pulseaudio --start

1. find your device by:
See: https://wiki.archlinux.org/index.php/PulseAudio/Examples

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

To get information of parameters, 


<a id="top"/> 
