from keras.engine import Layer, InputSpec
from keras.layers import Flatten, Permute
import tensorflow as tf
import keras.backend as K

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k, input_shape[2]

    def call(self, inputs, **kwargs):

        inputs = K.expand_dims(inputs, 2)
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, (0, 3, 2, 1))
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        top_k = Permute((2, 3, 1))(top_k)
        # return flattened output
        # return Flatten()(top_k)
        # top_k = Flatten()(top_k)

        '''
        Entry Inputs: (?, ?, 128)
        After Inputs: (?, ?, 1, 128)
        Output shape: (?, ?, 1, 128)
        Output squeeze shape: (?, ?, 128)
        K max pooling shape: (None, 150, 128)
        '''

        return K.squeeze(top_k, axis=1)
        # return K.expand_dims(Flatten()(top_k), 2)