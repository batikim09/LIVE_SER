#!/usr/bin/env python
import pyaudio
import wave
import audioop
import webrtcvad
import argparse
import sys
import os
import numpy as np
from decoding import Decoder 
import wave
import sys
import struct
from datetime import datetime

from helper import *


def listup_devices():
    p = pyaudio.PyAudio()
    #list up devices
    for i in range(p.get_device_count()):
        log(str(p.get_device_info_by_index(i)))

def find_device_id(name):
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            infor = "Input Device id " + str(i) + " - " +  str(p.get_device_info_by_host_api_device_index(0, i).get('name')) + " - ch: " + str(p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) + " sr: " + str(p.get_device_info_by_host_api_device_index(0, i).get('defaultSampleRate'))
            
        if name in p.get_device_info_by_host_api_device_index(0, i).get('name'):
            log(str( name + " is found and will be used as an input device."))
            return i
    log( "There is no such a device named " + name)
    return -1

def vad_result(task_outputs, predict_mode, logger = None):
    logs = ""
    for output in task_outputs:
        if predict_mode != 2:
            output_str = "\t" + str(output)
        else:
            output_str = ""
            for p in output:
                output_str = output_str + "\t" + str(p) 
        logs += output_str
    
    logs = logs[1:]
    log(logs, logger)

def no_vad_result(tasks, predict_mode, logger = None):
    logs = ""
    for num_classes in tasks:
        if predict_mode != 2:
            output_str = '\t-1.'
        else:
            output_str = ""
            for p in range(num_classes):
                output_str = output_str + "\t-1."
        logs += output_str
    
    logs = logs[1:]
    log(logs, logger)

def predict_frame(dec, frames, args, g_min_max = None, save = False):
    
    results = dec.predict(frames, g_min_max = g_min_max, feat_mode = args.feat_mode, feat_dim = args.feat_dim)
    
    if args.predict_mode == 0:
        task_outputs = dec.returnDiff(results)
    elif args.predict_mode == 1:
        task_outputs = dec.returnLabel(results)
    else:
        task_outputs = dec.returnClassDist(results)
    return task_outputs

def predict_file(dec, pyaudio, path, frames, args, rate = 16000, format = pyaudio.paInt16, g_min_max = None, save = False):
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.get_sample_size(format))
    wf.setframerate(rate)
    #this code works for only ...
    wf.writeframes(b''.join(frames))
    wf.close()

    results = dec.predict_file(path, g_min_max = g_min_max, feat_mode = args.feat_mode, feat_dim = args.feat_dim)
    
    if save == False:
        os.remove(path)

    if args.predict_mode == 0:
        task_outputs = dec.returnDiff(results)
    elif args.predict_mode == 1:
        task_outputs = dec.returnLabel(results)
    else:
        task_outputs = dec.returnClassDist(results)
    return task_outputs

def ser(args):

    if args.save:
        save = True
    else:
        save = False
    
    if args.log_file:
        logger = open(args.log_file, "w")
    else:
        logger = open(args.wave + ".vad.csv", "w")
            
    tasks = []
    
    for task in args.tasks.split(","):
        tasks.append(int(task.split(":")[1]))
        
    #audio device setup
    format = pyaudio.paInt16
    
    n_channel = args.n_channel
    sample_rate = args.sample_rate
    frame_duration = args.frame_duration
    frame_len = int(sample_rate * (frame_duration / 1000.0))
    chunk = int(frame_len/ args.n_channel)

    vad_mode = args.vad_mode
    
    log("frame_len: %d" % frame_len)
    log("chunk size: %d" % chunk)

    #feature extraction setting
    min_voice_frame_len = frame_len * (args.vad_duration / frame_duration)
    log("minimum voice frame length: %d" % min_voice_frame_len)
    feat_path = args.feat_path
    tmp_data_path = "./tmp"
    try:
        os.mkdir(tmp_data_path)
    except:
        log('Data folder already exists')
    

    #initialise vad
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    #automatic gain normalisation
    if args.g_min and args.g_max:
        g_min_max = (args.g_min, args.g_max)
    else:
        g_min_max = None

    if args.model_file:
        #initialise model
        if args.stl:
           dec = Decoder(model_file = args.model_file, elm_model_files = args.elm_model_file, context_len = args.context_len, max_time_steps = args.max_time_steps, tasks = args.tasks, sr = args.sample_rate)
        else:
           dec = Decoder(model_file = args.model_file, elm_model_files = args.elm_model_file, context_len = args.context_len, max_time_steps = args.max_time_steps, tasks = args.tasks, stl = False, sr = args.sample_rate)
            
    p = pyaudio.PyAudio()

    #open file (offline mode)
    if args.wave:
        f = wave.open(args.wave)

        if args.play:
            s = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    else:
        log("no input wav file! Starting a live mode.")
        #open mic
        if args.device_id is None:
            args.device_id = find_device_id("pulse")
        if args.device_id == -1:
            log("There is no default device!, please check the configuration")
            sys.exit(-1)
            
        #open mic
        f = p.open(format = format, channels = n_channel,rate = sample_rate,input = True, input_device_index = args.device_id,frames_per_buffer = chunk)

    log("---Starting---")

    is_currently_speech = False
    total_frame_len = 0
    frames_16i = ''
    frames_np = []
    prev_task_outputs = None
    speech_frame_len = 0
    total_frame_count = 0

    #read first frame
    while True:
        
        if args.wave:
            data = f.readframes(chunk)
        else:
            data = f.read(chunk)
            
        if data == '':
            break

        #play stream
        if args.play:
            s.write(data)

        #check gain
        mx = audioop.max(data, 2)
        #vad
        try:
            is_speech = vad.is_speech(data, sample_rate)
        except:
            log("end of speech")
            break

        if mx < args.min_energy:
            is_speech = 0
        
        if args.gain:
            log(str('gain: %d, vad: %d' % (mx, is_speech)))   
        
        if is_speech == 1:
            speech_frame_len = speech_frame_len + chunk #note chunk is a half of frame length.

        if args.save:
            if frames_16i == '': 
                frames_16i = data
            else:
                frames_16i = frames_16i + data
        
        frames_np.append(np.fromstring(data, dtype=np.int16))
        
        total_frame_len = total_frame_len + chunk

        if total_frame_len > min_voice_frame_len:       
             
            if args.model_file and float(speech_frame_len)/total_frame_len > args.speech_ratio:
                
                if args.save:
                    outputs = predict_file(dec, p, tmp_data_path + "/" + str(datetime.now()) + '.wav', frames_16i, args, g_min_max = g_min_max, save = save)
                else:
                    frames_np = np.hstack(frames_np)
                    outputs = predict_frame(dec, frames_np, args, g_min_max = g_min_max)
                    
                vad_result(outputs, args.predict_mode, logger)
            else:
                no_vad_result(tasks, args.predict_mode, logger)
            
            #initialise
            total_frame_len = 0
            speech_frame_len = 0
            frames_16i = ''
            frames_np = []

        total_frame_count = total_frame_count + 1
        if total_frame_count % 100 == 0:
            log(str("total frame: %d" %( total_frame_count)))

    log("---done---")

    if args.wave:
        f.close()
    if args.play:
        s.stop_stream()
        s.close()
    if args.log_file:
        logger.close()

    p.terminate()   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d_id", "--device_id", dest= 'device_id', type=int, help="device id for microphone", default=None)
    
    #options for VAD
    parser.add_argument("-sr", "--sample_rate", dest= 'sample_rate', type=int, help="number of samples per sec, only accept [8000|16000|32000]", default=16000)
    parser.add_argument("-ch", "--n_channel", dest= 'n_channel', type=int, help="number of channels", default=1)
    parser.add_argument("-fd", "--frame_duration", dest= 'frame_duration', type=int, help="a duration of a frame msec, only accept [10|20|30]", default=20)
    parser.add_argument("-vm", "--vad_mode", dest= 'vad_mode', type=int, help="vad mode, only accept [0|1|2|3], 0 more quiet 3 more noisy", default=0)
    parser.add_argument("-vd", "--vad_duration", dest= 'vad_duration', type=int, help="minimum length(ms) of speech for emotion detection", default=1000)
    parser.add_argument("-me", "--min_energy", dest= 'min_energy', type=int, help="minimum energy of speech for emotion detection", default=100)
    parser.add_argument("-wav", "--wave", dest= 'wave', type=str, help="wave file (offline mode)")
    
    #automatic gain normalisation
    parser.add_argument("-g_min", "--gain_min", dest= 'g_min', type=float, help="min value of automatic gain normalisation")
    parser.add_argument("-g_max", "--gain_max", dest= 'g_max', type=float, help="max value of automatic gain normalisation")
    parser.add_argument("-s_ratio", "--speech_ratio", dest= 'speech_ratio', type=float, help="speech ratio", default=0.3)

    #options for Model
    parser.add_argument("-fp", "--feat_path", dest= 'feat_path', type=str, help="temporay feat path", default='./temp.csv')
    parser.add_argument("-md", "--model_file", dest= 'model_file', type=str, help="keras model path")
    parser.add_argument("-elm_md", "--elm_model_file", dest= 'elm_model_file', type=str, help="elm model_file")
    parser.add_argument("-c_len", "--context_len", dest= 'context_len', type=int, help="context window's length", default=10)
    parser.add_argument("-m_t_step", "--max_time_steps", dest= 'max_time_steps', type=int, help="maximum time steps for DNN", default=500)
    
    parser.add_argument("-log", "--log_file", dest= 'log_file', type=str, help="log")
    #parser.add_argument("-n_class", "--n_class", dest= 'n_class', type=int, help="number of class", default=2)
    parser.add_argument("-tasks", "--tasks", dest = "tasks", type=str, help ="tasks (arousal:2,valence:2)", default='emotion_category')
    parser.add_argument("-p_mode","--predict", dest = 'predict_mode', type=int, help=("0 = diff, 1 = classification, 2 = distribution"), default = 2)
    parser.add_argument("-f_mode","--feat_mode", dest = 'feat_mode', type=int, help=("0 = lspec, 1 = raw wav"), default = 0)
    parser.add_argument("-f_dim","--feat_dim", dest = 'feat_dim', type=int, help=("feature dimension (# spec for lspec or mspec"), default = 80)
    parser.add_argument("--stl", help="only for single task learning model", action="store_true")
    parser.add_argument("--save", help="save detected voice segments", action="store_true")
    parser.add_argument("--play", help="play a given audio file in real-time", action="store_true")
    parser.add_argument("--gain", help="show gains of the selected microphone", action="store_true")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        listup_devices()       
        sys.exit(1)
    
    print("args: " + str(args))
    ser(args)
