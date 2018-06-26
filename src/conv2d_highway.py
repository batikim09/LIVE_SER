import numpy

from keras.layers import *
from keras.layers.convolutional import _Conv
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

class Conv2DHighway(_Conv):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DHighway, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(Conv2DHighway, self).get_config()
        config.pop('rank')
        return config

    def build(self, input_shape):

        super(Conv2DHighway, self).build(input_shape)
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel_gate = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel_gate',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.bias_gate = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias_gate',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias_gate = None
            
    def get_output_shape_for(self, input_shape):
        super(Conv2DHighway, self).get_output_shape_for(input_shape)

    def call(self, inputs, mask=None):
        # compute channels_firste candidate hidden state
         # Arguments
        '''
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.
        '''
        transform = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            transform = K.bias_add(
                transform,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            transform = self.activation(transform)

        transform_gate = K.conv2d(
                inputs,
                self.kernel_gate,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            transform = K.bias_add(
                transform,
                self.bias_gate,
                data_format=self.data_format)
        
        transform_gate = K.sigmoid(transform_gate)
       
        carry_gate = 1.0 - transform_gate

        return transform * transform_gate + inputs * carry_gate

