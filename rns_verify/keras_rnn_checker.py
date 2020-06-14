from keras.models import Sequential
from keras.layers import Dense, Activation
from rns_verify.rnn import RNN


def construct_rnn_from_keras_rnn(keras_model, built_myself=False):
    # TODO: What happened to the bias?
    # TODO: What if I have more than one Dense layer?
    recurrent_layer_idx = 0
    hidden_output_layer_idx = 1
    if built_myself:
        recurrent_layer_idx = 2
        hidden_output_layer_idx = 3
    recurrent_layer = keras_model.layers[recurrent_layer_idx].get_weights()
    input_hidden = recurrent_layer[0]
    hidden_hidden = recurrent_layer[1]
    hidden_bias = recurrent_layer[2]
    hidden_output = keras_model.layers[hidden_output_layer_idx].get_weights()[0]
    return RNN(input_hidden, hidden_hidden, hidden_output, hidden_bias)


def construct_keras_ffnn_from_rnn(rnn, sequence_length):
    ffnn = rnn.unroll(sequence_length)
    model = Sequential()
    ffnn_input_hidden = ffnn.get_input_hidden()
    ffnn_input_shape = ffnn_input_hidden.shape
    model.add(Dense(units=ffnn_input_shape[1], input_dim=ffnn_input_shape[0], weights=[ffnn_input_hidden], use_bias=False))
    model.add(Activation('relu'))
    for hidden_layer in ffnn.get_hidden_layers():
        ffnn_hidden_shape = hidden_layer.shape
        model.add(Dense(units=ffnn_hidden_shape[1], input_dim=ffnn_hidden_shape[0], weights=[hidden_layer], use_bias=False))
        model.add(Activation('relu'))

    ffnn_output = ffnn.get_hidden_output()
    ffnn_output_shape = ffnn_output.shape
    model.add(Dense(units=ffnn_output_shape[1], input_dim=ffnn_output_shape[0], weights=[ffnn_output], use_bias=False))
    return model
