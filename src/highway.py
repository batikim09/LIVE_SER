from keras.engine import InputSpec
from keras.layers import Dense
from keras.layers.wrappers import Wrapper, TimeDistributed
 
 
class Highway(Wrapper):
    def __init__(self, layer, gate=None, **kwargs):
        self.supports_masking = True
        self.gate = gate
        super(Highway, self).__init__(layer, **kwargs)
 
    def build(self, input_shape=None):
        assert len(input_shape) in [2, 3]
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_output_dims = input_shape[-1]
 
        if self.gate is None:
            gate = Dense(nb_output_dims, activation='sigmoid')
            if len(input_shape) == 3:
                gate = TimeDistributed(gate)
            self.gate = gate
 
        super(Highway, self).build(input_shape)
 
    def get_output_shape_for(self, input_shape):
        assert self.layer.get_output_shape_for(input_shape) == input_shape
        assert self.gate.get_output_shape_for(input_shape) == input_shape
        return input_shape
 
    def call(self, x, mask=None):
        return self.layer(x) * self.gate(x) + x * (1 - self.gate(x))