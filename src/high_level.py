import numpy as np
import tensorflow as tf
from elm import ELM
from keras.utils import np_utils

def generate_temporal_labels(multiTasks, Y_train, Y_test, Y_valid, max_t_steps):
    dictForLabelsTemporalTest = {}
    dictForLabelsTemporalValid = {}
    dictForLabelsTemporalTrain = {}
    dictForLabelsTest = {}
    dictForLabelsValid = {}
    dictForLabelsTrain = {}

    for task, classes, idx in multiTasks:

        if Y_train != None:
            dictForLabelsTemporalTrain[task] = time_distributed_label(np_utils.to_categorical(Y_train[:,idx], classes), max_t_steps)
            dictForLabelsTrain[task] = np_utils.to_categorical(Y_train[:,idx], classes)
        if Y_test != None:
            dictForLabelsTemporalTest[task] = time_distributed_label(np_utils.to_categorical(Y_test[:,idx], classes), max_t_steps)
            dictForLabelsTest[task] = np_utils.to_categorical(Y_test[:,idx], classes)
        if Y_valid != None:    
            dictForLabelsTemporalValid[task] = time_distributed_label(np_utils.to_categorical(Y_valid[:,idx], classes), max_t_steps)
            dictForLabelsValid[task] = np_utils.to_categorical(Y_valid[:,idx], classes)
            
    return dictForLabelsTemporalTest, dictForLabelsTemporalValid, dictForLabelsTemporalTrain, dictForLabelsTest, dictForLabelsValid, dictForLabelsTrain


def time_distributed_label(label, max_t_steps):
    nb_class = label.shape[1]
    nb_samples = label.shape[0]
    dictForLabelsTime = np.zeros((nb_samples, max_t_steps, nb_class))
    for i in range(nb_samples):
        for t in range(max_t_steps):
            dictForLabelsTime[i, t, ] = label[i]
    return dictForLabelsTime

def high_level_feature_mtl(predictions, threshold = 0.3, stl = False, main_task_id = -1):
    results = []
    total_feat = 0
    total_samples = 0
    if stl == True:
        total_samples = predictions.shape[0]
        result, nb_classes = high_level_feature_task(predictions, threshold)
        return result
    else:
        num_tasks = len(predictions)    
        total_samples = predictions[0].shape[0]
        for task_id in range(num_tasks):
            result, nb_classes = high_level_feature_task(predictions[task_id], threshold)
            results.append((result, total_feat, total_feat + nb_classes * 4))
            total_feat = total_feat + nb_classes * 4

    #high level representation of all tasks
    if main_task_id == -1:
        feature_vecs = np.zeros((total_samples, total_feat))
        for utt_id in range(total_samples):
            for task_id in range(num_tasks):
                result = results[task_id]
                feature_vecs[utt_id][result[1]:result[2]] = result[0][utt_id]
    else:
        feature_vecs = results[main_task_id][0]
    return feature_vecs

def high_level_feature_task(predictions, threshold):
    batch_size = predictions.shape[0]
    nb_classes = predictions.shape[2]
    results = np.zeros((batch_size, nb_classes * 4))
    for utt_idx in range(batch_size):
        results[utt_idx] = high_level_feature(predictions[utt_idx], threshold)
    return results, nb_classes

#prediction result for each utturance(time * classes)
def high_level_feature(pred, threshold):
    max_idx = np.argmax(pred, 0)
    min_idx = np.argmin(pred, 0)
    nb_classes = pred.shape[1]
    max_scores = np.zeros((nb_classes))
    min_scores = np.zeros((nb_classes))
    for idx in range(len(max_idx)):
        max_scores[idx] = pred[max_idx[idx]][idx]    
    for idx in range(len(min_idx)):
        min_scores[idx] = pred[min_idx[idx]][idx]
    avg_scores = np.mean(pred, 0)    
    sums = np.sum(pred, 0) + 0.000001
    over = sum(pred > threshold)
    portions = over * avg_scores / sums
    results = np.zeros((nb_classes * 4))
    feats = (max_scores, min_scores, avg_scores, portions)
    for feat_idx in range(4):
        results[feat_idx * nb_classes : (feat_idx + 1) * nb_classes] = feats[feat_idx]
    return results
