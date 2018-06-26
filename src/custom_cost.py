from keras import backend as K
import itertools
import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(2016)

gamma = 2.
alpha = .25
nb_class = 4
weights = []

def init_categorical_focal_loss(i_nb_class, i_gamma = 2., i_alpha = .25 ):
    gamma = i_gamma
    alpha = i_alpha
    nb_class = i_nb_class
    
def categorical_focal_loss(y_true, y_pred):
    if(K.backend()=="tensorflow"):
        import tensorflow as tf
        pt = y_pred
        return -K.sum(alpha * y_true * K.pow(1. - pt, gamma) * K.log(pt)) / nb_class
    if(K.backend()=="theano"):
        import theano.tensor as T
        pt = y_pred
        return -K.sum(alpha * y_true * K.pow(1. - pt, gamma) * K.log(pt)) / nb_class

def init_w_categorical_crossentropy(i_weights):
    nb_cl = len(i_weights)
    weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in i_weights.items():
        weights[0][class_idx] = class_weight
        weights[class_idx][0] = class_weight

def w_categorical_crossentropy(y_true, y_pred):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

class WeightedCategoricalCrossEntropy(object):

  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
      self.weights[0][class_idx] = class_weight
      self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_true, y_pred) * final_mask


    
class CategoricalFocalLoss(object):
    def __init__(self, nb_class, gamma=2., alpha=.25):
        self.gamma = gamma
        self.alpha = alpha
        self.nb_class = nb_class
        self.__name__ = 'categorical_focal_loss'
    def __call__(self, y_true, y_pred):
        return self.focal_loss_fixed(y_true, y_pred)
    
    def focal_loss_fixed(self, y_true, y_pred):
        if(K.backend()=="tensorflow"):
            import tensorflow as tf
            pt = y_pred
            return -K.sum(self.alpha * y_true * K.pow(1. - pt, self.gamma) * K.log(pt)) / self.nb_class
        if(K.backend()=="theano"):
            import theano.tensor as T
            pt = y_pred
            return -K.sum(self.alpha * y_true * K.pow(1. - pt, self.gamma) * K.log(pt)) / self.nb_class
    #def get_shape(self):
    #    return 0
class BinaryFocalLoss(object):
    def __init__(self, gamma=2, alpha=2):
        self.gamma = gamma
        self.alpha = alpha
        self.__name__ = 'binary_focal_loss'
    def __call__(self, y_true, y_pred):
        return self.focal_loss_fixed(y_true, y_pred)
    
    def focal_loss_fixed(self, y_true, y_pred):
        if(K.backend()=="tensorflow"):
            import tensorflow as tf
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(self.alpha * K.pow(1. - pt, self.gamma) * K.log(pt))
        if(K.backend()=="theano"):
            import theano.tensor as T
            pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(self.alpha * K.pow(1. - pt, self.gamma) * K.log(pt))