"""Cartpole DQN models."""
from keras.layers import Dense, Layer, Input
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K

# pylint: disable=line-too-long


class RNNCell(Layer):
    """Wrapped to fix call signature for super cell class."""
    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build the layer."""
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units),
                                      name='kernel',
                                      initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer='identity',
            regularizer=None,
            constraint=None)
        self.bias = self.add_weight(shape=(self.units,),
                                    name='bias',
                                    initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        """Delegate assuming inputs[1] is states."""
        assert len(inputs) == 2
        state, hm1 = inputs
        h = K.dot(state, self.kernel) + K.dot(hm1, self.recurrent_kernel)
        # h = K.dot(state, self.kernel)
        h = K.bias_add(h, self.bias)
        return K.relu(h)

    def compute_output_shape(self, input_shape):
        """Compute the output shape list."""
        return input_shape[1]


def build_model():
    """Build a stateful simple RNN model."""
    action_space_size = 2
    hidden_size = 16
    observation_space_shape = 3
    state_in = Input(shape=(observation_space_shape,), name='state_in')
    hidden_in = Input(shape=(hidden_size,), name='hidden_in')

    latent = RNNCell(hidden_size)([state_in, hidden_in])
    # Verification tool currently supports only one dense layer for RNN.
    # self.model.add(Dense(hidden_size, activation='relu'))
    # qvals = Dense(hidden_size, activation='relu')(latent)
    qvals = Dense(action_space_size, activation='linear')(latent)
    model = Model([state_in, hidden_in], [qvals, latent])
    return model
