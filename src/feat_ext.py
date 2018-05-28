import numpy as np
import librosa
from sklearn.decomposition import PCA
from librosa.util import buf_to_float
import sys

class FeatExt(object):
    def __init__(self, min_max):

        if min_max:
            self.min = min_max[0]
            self.max = min_max[1]
            self.auto_gain_control = False
        else:
            self.min = sys.float_info.max
            self.max = -1.0
            self.auto_gain_control = True

    def gain_norm(self, frames):
        if self.auto_gain_control:
            temp_min = np.min(frames)
            temp_max = np.max(frames)
            if temp_min < self.min:
                self.min = temp_min
                #debug
                print("new min", self.min)
            if temp_max > self.max:
                self.max = temp_max
                #debug
                print("new max", self.max)
            frames = (frames - self.min)/(self.max - self.min)
        else:            
            frames = (frames - self.min)/(self.max - self.min)
            
        return frames

    def extract_feat_frame(self, frames, mode = 0, file = None, sr = 16000, n_fft=512, hop_length=512, n_mels=40, fmax= 8000, convert_16I_to_32F = True):
        
        #convert 16 bit integer to 32 float
        if convert_16I_to_32F:
            n_frames = []
            for frame in frames:
                n_frames.append(buf_to_float(frame, 2, np.float32))
            n_frames = np.concatenate(n_frames)
            frames = np.ascontiguousarray(n_frames, np.float32)
            
        frames = self.gain_norm(frames)
            
        if mode == 0:
            return self.extract_melspec_frame(frames, file = file, n_mels = n_mels, sr = sr)
        elif mode == 1:
            return self.extract_wav_frame(frames, file = file)
        elif mode == 2:
            return self.extract_log_spectrogram_frame(frames, file = file, n_mels = n_mels, sr = sr)
        else:
            print("non-supported feature type")

    def extract_feat_file(self, path, mode = 0, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000):
        
        if mode == 0:
            return self.extract_melspec_file(path, file = file, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax)
        elif mode == 1: 
            return self.extract_wav_file(path, file = file)
        elif mode == 2:
            return self.extract_log_spectrogram_file(path, file = file, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax)
        else:
            print("non-supported feature type")

    def extract_melspec_frame(self, frames, file = None, sr = 16000, n_fft=512, hop_length=512, n_mels=40, fmax= 8000):

        mel = librosa.feature.melspectrogram(y=frames, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax= fmax) 
        mel = mel.T
        
        if file != None:
            np.savetxt(file, mel, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
            
        return mel

    def extract_wav_frame(self, frames, file = None, sr = 16000):

        
        r = np.zeros((len(frames), 1))
        r[:, 0] = frames[:]

        
        if file != None:
            np.savetxt(file, r, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
            
        return r


    def extract_log_spectrogram_frame(self, frames, file = None, sr = 16000, n_fft=512, hop_length=512):

        #spec = librosa.feature.logfsgram(y=frames, sr=sr, S=None, n_fft=n_fft, hop_length=hop_length)
        spec = np.abs(librosa.stft(frames, n_fft = n_fft))
        log_spec = librosa.amplitude_to_db(spec**2)
        log_spec = log_spec.T

        if file != None:
            np.savetxt(file, log_spec, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')

        return log_spec 

    def extract_pca_whitenining(self, frames, pca_components = 60):
        pca = PCA(n_components=pca_components, whiten = True)
        return pca.fit_transform(frames)

    def extract_pca_logspec_frame(self, frames, file = None, sr = 16000, n_fft=512, hop_length=512, pca_components = 60):
        spec = self.extract_log_spectrogram_frame(frames, file = None, sr = sr, n_fft=n_fft, hop_length=hop_length)
        pca_spec = self.extract_pca_whitenining(spec, pca_components= pca_components)

        if file != None:
            np.savetxt(file, pca_spec, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')

        return pca_spec

    def extract_melspec_file(self, path, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000):

        y, sr = librosa.load(path)
        y = self.gain_norm(y)

        mel = self.extract_melspec_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, fmax = fmax)
        return mel

    def extract_log_spectrogram_file(self, path, file = None, n_fft=512, hop_length=512, n_mels=40, fmax= 8000):

        y, sr = librosa.load(path)
        y = self.gain_norm(y)

        spec = self.extract_log_spectrogram_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length )
        return spec

    def extract_pca_logspec_file(self, path, file = None, n_fft=512, hop_length=512, fmax= 8000, pca_components = 60):

        y, sr = librosa.load(path)
        y = self.gain_norm(y)

        spec = self.extract_pca_logspec_frame(y, file = file, sr = sr, n_fft = n_fft, hop_length = hop_length, pca_components =pca_components)
        return spec

    def extract_wav_file(self, path, file = None):

        y, sr = librosa.load(path)
        y = self.gain_norm(y)

        r = np.zeros((len(y), 1))
        r[:, 0] = y[:]

        if file != None:
            np.savetxt(file, r, fmt='%.8e', delimiter=';', newline='\n', header='', footer='')
        return r

