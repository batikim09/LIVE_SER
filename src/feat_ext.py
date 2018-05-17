import numpy as np
import librosa
from sklearn.decomposition import PCA
from librosa.util import buf_to_float
import sys

#global variables for min-max
max = -1.0
min = sys.float_info.max

def gain_norm(frames, min_max):
    if min_max:
        min = min_max[0]
        max = min_max[1]
        frames = (frames - min)/(max - min)
    else: #auto gain control
        temp_min = np.min(frames)
        temp_max = np.max(frames)
        global min
        global max
        if temp_min < min:
            min = temp_min
            #debug
            print("new min", min)
        if temp_max > max:
            max = temp_max
            #debug
            print("new max", max)
        frames = (frames - min)/(max - min)
    return frames

def extract_feat_frame(frames, mode = 0, file = None, sr = 16000, n_fft=512, hop_length=512, n_mels=40, fmax= 8000, min_max = None, convert_16I_to_32F = True):
    
    #convert 16 bit integer to 32 float
    if convert_16I_to_32F:
        n_frames = []
        for frame in frames:
            n_frames.append(buf_to_float(frame, 2, np.float32))
        n_frames = np.concatenate(n_frames)
        frames = np.ascontiguousarray(n_frames, np.float32)
        
    frames = gain_norm(frames, min_max)
        
    if mode == 0:
        return extract_melspec_frame(frames, file = file, n_mels = n_mels, sr = sr)
    elif mode == 1:
        return extract_wav_frame(frames, file = file)
    elif mode == 2:
        return extract_log_spectrogram_frame(frames, file = file, n_mels = n_mels, sr = sr)
    else:
        print("non-supported feature type")

def extract_feat_file(path, mode = 0, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000, min_max = None):
    
    if mode == 0:
        return extract_melspec_file(path, file = file, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax, min_max = min_max)
    elif mode == 1: 
        return extract_wav_file(path, file = file, min_max = min_max)
    elif mode == 2:
        return extract_log_spectrogram_file(path, file = file, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax, min_max = min_max)
    else:
        print("non-supported feature type")

def extract_melspec_frame(frames, file = None, sr = 16000, n_fft=512, hop_length=512, n_mels=40, fmax= 8000):

    mel = librosa.feature.melspectrogram(y=frames, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax) 
    mel = mel.T
    
    if file != None:
        np.savetxt(file, mel, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
        
    return mel

def extract_wav_frame(frames, file = None, sr = 16000):

    
    r = np.zeros((len(frames), 1))
    r[:, 0] = frames[:]

    
    if file != None:
        np.savetxt(file, r, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
        
    return r


def extract_log_spectrogram_frame(frames, file = None, sr = 16000, n_fft=512, hop_length=512):

    #spec = librosa.feature.logfsgram(y=frames, sr=sr, S=None, n_fft=n_fft, hop_length=hop_length)
    spec = np.abs(librosa.stft(frames, n_fft = n_fft))
    log_spec = librosa.amplitude_to_db(spec**2)
    log_spec = log_spec.T

    if file != None:
        np.savetxt(file, log_spec, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')

    return log_spec 

def extract_pca_whitenining(frames, pca_components = 60):
    pca = PCA(n_components=pca_components, whiten = True)
    return pca.fit_transform(frames)

def extract_pca_logspec_frame(frames, file = None, sr = 16000, n_fft=512, hop_length=512, pca_components = 60):
    spec = extract_log_spectrogram_frame(frames, file = None, sr = sr, n_fft=n_fft, hop_length=hop_length)
    pca_spec = extract_pca_whitenining(spec, pca_components= pca_components)

    if file != None:
        np.savetxt(file, pca_spec, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')

    return pca_spec

def extract_melspec_file(path, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000, min_max = None):

    y, sr = librosa.load(path)
    y = gain_norm(y, min_max)

    mel = extract_melspec_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, fmax = fmax)
    return mel

def extract_log_spectrogram_file(path, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000, min_max = None):

    y, sr = librosa.load(path)
    y = gain_norm(y, min_max)

    spec = extract_log_spectrogram_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length )
    return spec

def extract_pca_logspec_file(path, file = None, n_fft=512, hop_length=512, fmax= 8000, pca_components = 60, min_max = None):

    y, sr = librosa.load(path)
    y = gain_norm(y, min_max)

    spec = extract_pca_logspec_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length, pca_components =pca_components)
    return spec

def extract_wav_file(path, file = None, min_max = None):

    y, sr = librosa.load(path)
    y = gain_norm(y, min_max)

    r = np.zeros((len(y), 1))
    r[:, 0] = y[:]

    if file != None:
        np.savetxt(file, r, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
    return r

