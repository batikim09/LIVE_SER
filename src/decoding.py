from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
import sys
import argparse

from keras import backend as K
from keras.models import Model
from feat_ext import *
from elm import ELM
from high_level import *

from highway import Highway 
from conv2d_highway import Conv2DHighway
from conv1d_highway import Conv1DHighway
from conv3d_highway import Conv3DHighway
from custom_cost import *
from custom_metric import *



class Decoder(object):
    def __init__(self, model_file = './model.h5', elm_model_files = None, feat_path = './temp.csv', context_len = 5, max_time_steps = 300, elm_hidden_num = 50, stl = True, elm_main_task_id = -1, sr = 16000, tasks = 'arousal:2,valence:2', min_max = None, seq2seq = False):
        
        self.stl = stl
        self.model = self.model = keras.models.load_model(model_file, custom_objects={'Conv3DHighway': Conv3DHighway, 'Conv2DHighway': Conv2DHighway, 'Conv1DHighway': Conv1DHighway, 'Highway': Highway, 'w_categorical_crossentropy': w_categorical_crossentropy, 'categorical_focal_loss': categorical_focal_loss, 'f1': f1, 'precision': precision, 'recall': recall})
        self.seq2seq = seq2seq

        self.elm_model_files = elm_model_files
        self.sess = tf.Session()
        self.elm_model = []
        self.tasks = []
        self.tasks_names = []
        self.total_high_level_feat = 0

        self.feat_ext = FeatExt(min_max)

        #setting multi-task
        for task in tasks.split(","):
            print("task: ", task)
            task_n_class = task.split(':') 
            self.tasks.append(int(task_n_class[1]))
            self.tasks_names.append(task_n_class[0])
            self.total_high_level_feat = self.total_high_level_feat + int(task_n_class[1])            
        
        #seeting an elm model for a post-classifier
        if self.elm_model_files != None:
            print("elm model is loaded")
            elm_tasks = elm_model_files.split(',')
            if len(elm_tasks) == len(self.tasks):
                print("#tasks: ", len(self.tasks))

                for i in range(0, len(self.tasks)):
                    elm_model_task = ELM(self.sess, 1, self.total_high_level_feat * 4, elm_hidden_num, self.tasks[i], task = self.tasks_names[i])
                    elm_path = elm_tasks[i] 
                    elm_model_task.load(elm_path)

                    self.elm_model.append(elm_model_task)   
                self.elm_hidden_num = elm_hidden_num
                self.elm_main_task_id = elm_main_task_id
            else:
                print("mismatch between tasks and elm models")
                exit()

        self.sr = sr
        self.feat_path = feat_path
        self.context_len = context_len
        self.max_time_steps = max_time_steps   
        self.model.summary()

    #predict frames
    def predict(self, frames, feat_mode = 0, feat_dim = 80, three_d = False):
        feat = self.feat_ext.extract_feat_frame(frames, mode = feat_mode, file = self.feat_path, n_mels = feat_dim, sr = self.sr)
        temporal_feat = self.build_temporal_feat(feat, three_d)
        return self.temporal_predict(temporal_feat)

    #predict frames in a file
    def predict_file(self, input_file, feat_mode = 0, feat_dim = 80, three_d = False):
        feat = self.feat_ext.extract_feat_file(input_file, mode = feat_mode, file = self.feat_path, n_mels = feat_dim)
        temporal_feat = self.build_temporal_feat(feat, three_d)
        return self.temporal_predict(temporal_feat)
    
    #predict frames in a long file
    def predict_long_file(self, input_file, feat_mode = 0, feat_dim = 80, three_d = False):
        feat = self.feat_ext.extract_feat_file(input_file, mode = feat_mode, file = self.feat_path, n_mels = feat_dim)

        n_turns = feat.shape[0] / self.max_time_steps
        result = []
        #devide the total frames by several turns; each turn has the maximum number of time steps.
        for i in range(0, n_turns):
            start = i * self.max_time_steps
            end = (i + 1) * self.max_time_steps
            temporal_feat = self.build_temporal_feat(feat[start:end], three_d)
            result.append(self.temporal_predict(temporal_feat))
        return result

    #prediction using temporal features
    def temporal_predict(self, temporal_feat):  
        print("temporal feat shape: ", temporal_feat.shape)
        #print(temporal_feat[0,0,0,0,0])
        predictions = self.model.predict(temporal_feat)

        if self.elm_model_files == None:
            preds = []
            #print("prediction: ", str(predictions))
            for i in range(0, len(self.tasks)): 
                #preds.append(predictions[i][0])
                preds.append(predictions[i])
        else:
            feat_test = high_level_feature_mtl(predictions, threshold = 0.3, stl = self.stl, main_task_id = self.elm_main_task_id)
            preds = []
            #print("feat: ", str(feat_test))
            for i in range(0, len(self.tasks)):
                elm_predictions = self.elm_model[i].test(feat_test)
                #print("shape", elm_predictions.shape)
                preds.append(elm_predictions)
        
        return preds

    #compose a temporal feature structure
    def build_temporal_feat(self, input_feat, three_d = False):
        print("feature shape:", input_feat.shape)

        input_dim = input_feat.shape[1]
        max_t_steps = int(self.max_time_steps / self.context_len)

        if three_d:
            feat = np.zeros((1, 1, max_t_steps, self.context_len, input_dim))
        else:    
            feat = np.zeros((1, max_t_steps, 1, self.context_len, input_dim))

        for t_steps in range(max_t_steps):
            if t_steps * self.context_len < input_feat.shape[0] - self.context_len:
                if input_feat.shape[1] != input_dim:
                    print('inconsistent dim')
                    break
                for c in range(self.context_len):
                    if three_d:
                        feat[0, 0, t_steps, c, ] = input_feat[t_steps * self.context_len + c]
                    else:
                        feat[0, t_steps, 0, c, ] = input_feat[t_steps * self.context_len + c]    
                        
        return feat

    #classification mode
    def returnLabel(self, result):
        labels = []
        
        print(result)

        #multi-tasks output format
        for task in result:
            
            if self.seq2seq:
                label = np.argmax(task, 1)
                most_frequent = np.bincount(label).argmax()
                print("most frequent label:", str(most_frequent))
                labels.append(most_frequent)
            else:
                label = np.argmax(task, 0)    
                print("label:", str(label))
                labels.append(label)
        
        #but results are always multi-tasks format
        return labels
    
    #distribution mode
    def returnClassDist(self, result):
        
        labels = []
        #multi-tasks output format
        print(result)

        for task in result:
            print("task result shape: ", task.shape)
            values = task.T

            class_avg_p = []
            for class_p in values:
                class_avg_p.append(np.mean(class_p))

            labels.append(class_avg_p)
        
        #but results are always multi-tasks format
        return labels
    
    #return the difference between probabilities of the first and last class.        
    def returnDiff(self, result):
        labels = []
        #multi-tasks output format
        print(result)

        for task in result:
            values = task.T
            label = values[-1] - values[0]
            avg = np.mean(label)
            labels.append(avg)
        
        #but results are always multi-tasks format
        return labels   

    def returnResult(self, results, mode = 2):
        
        if mode == 0:
            task_outputs = dec.returnDiff(results)
        elif mode == 1:
            task_outputs = dec.returnLabel(results)
        else:
            task_outputs = dec.returnClassDist(results)

        return task_outputs

    def returnSeqResult(self, results, mode = 2):
        task_outputs = []
        for result in results:
            if mode == 0:
                task_outputs.append(dec.returnDiff(result))
            elif mode == 1:
                task_outputs.append(dec.returnLabel(result))
            else:
                task_outputs.append(dec.returnClassDist(result))
        return task_outputs

def write_result(file_name, task_outputs):
    output = open(file_name, 'w')
    for task in task_outputs:
        for item in task:
            output.write( str(item) + '\t')
    output.close()
    
def write_seq_result(file_name, seq_task_outputs):
    output = open(file_name, 'w')
    for task_outputs in seq_task_outputs:    
        for task in task_outputs:
            for item in task:
                output.write( str(item) + '\t')
        output.write( '\n' )    
    output.close()

def write_named_seq_result(file_name, named_seq_task_outputs):
    output = open(file_name, 'w')
    for name, task_outputs in named_seq_task_outputs:
        output.write(name + '\t')
        for task in task_outputs:
            for item in task:
                output.write( str(item) + '\t')
        output.write( '\n' )    
    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wav", "--wave", dest= 'wave', type=str, help="wave file", default='./test.wav')
    parser.add_argument("-l_wav", "--long_wave", dest= 'long_wave', type=str, help="long wave file")
    parser.add_argument("-wav_list", "--wave_list", dest= 'wave_list', type=str, help="wave file list")
    parser.add_argument("-g_min", "--gain_min", dest= 'g_min', type=float, help="min value of automatic gain normalisation", default=-1.37686)
    parser.add_argument("-g_max", "--gain_max", dest= 'g_max', type=float, help="max value of automatic gain normalisation", default=1.55433)
    parser.add_argument("-md", "--model_file", dest= 'model_file', type=str, help="model file", default='./model.h5')
    parser.add_argument("-elm_md", "--elm_model_files", dest= 'elm_model_files', type=str, help="elm_model_file")
    parser.add_argument("-c_len", "--context_len", dest= 'context_len', type=int, help="context_len", default=5)
    parser.add_argument("-m_t_step", "--max_time_steps", dest= 'max_time_steps', type=int, help="max_time_steps", default=500)
    parser.add_argument("-tasks", "--tasks", dest = "tasks", type=str, help ="tasks (arousal:2,valence:2)", default='emotion_category')
    parser.add_argument("--default", help="default", action="store_true")
    parser.add_argument("--stl", help="stl", action="store_true")
    parser.add_argument("-p_mode","--predict_mode", dest = 'predict_mode', type=int, help=("0 = diff, 1 = classification, 2 = distribution"), default = 2)
    parser.add_argument("-f_mode","--feat_mode", dest = 'feat_mode', type=int, help=("0 = lspec, 1 = raw wav"), default = 0)
    parser.add_argument("-f_dim","--feat_dim", dest = 'feat_dim', type=int, help=("feature dimension (# spec for lspec or mspec"), default = 80)

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.g_min and args.g_max:
        g_min_max = (args.g_min, args.g_max)
    else:
        g_min_max = None

    if args.stl:
        dec = Decoder(model_file = args.model_file, elm_model_files = args.elm_model_files, feat_path = './temp.csv', context_len = args.context_len, max_time_steps = args.max_time_steps, tasks=args.tasks, min_max = g_min_max)
    else:
        dec = Decoder(model_file = args.model_file, elm_model_files = args.elm_model_files, feat_path = './temp.csv', context_len = args.context_len, max_time_steps = args.max_time_steps, tasks=args.tasks, stl = False, min_max = g_min_max)
    
    if args.long_wave:
        result = dec.predict_long_file(args.long_wave, g_min_max, feat_mode = args.feat_mode, feat_dim = args.feat_dim)
        output = args.long_wave + '.out.' + str(args.predict_mode) + "." + str(args.feat_mode) + '.csv'
        seq_task_outputs = dec.returnSeqResult(result, args.predict_mode)
                    
        write_seq_result(output, seq_task_outputs)
        print('batch decoding is done, total number of outputs: ', len(result))
        
    elif args.wave_list:
        inputs = open(args.wave_list, 'r')
        output = args.wave_list + '.out.' + str(args.predict_mode) + "." + str(args.feat_mode) + '.csv'
        named_seq_results = []
        for input in inputs:
            input = input.rstrip()
            result = dec.predict_file(input, g_min_max, feat_mode = args.feat_mode, feat_dim = args.feat_dim)
            print(result)
            task_outputs = dec.returnResult(result, args.predict_mode)
            named_seq_results.append((input, task_outputs))
        write_named_seq_result(output, named_seq_results)
    else:
        result = dec.predict_file(args.wave, feat_mode = args.feat_mode, feat_dim = args.feat_dim)
        output = args.wave + '.out.' + str(args.predict_mode) + "." + str(args.feat_mode) + '.csv'
        print(result)
        task_outputs = dec.returnResult(result, args.predict_mode)
        write_result(output, task_outputs)