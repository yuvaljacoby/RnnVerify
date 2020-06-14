import rns_verify.keras_rnn_checker
import keras.utils
import numpy as np

from rns_verify.constants import *
from rns_verify.ffnn import FFNN


class RNNAbstractor(object):
    """

    Attributes:
        rnn: Given RNN
        seq_len: Sequence Length
        abstraction_type: Abstraction Type
    """

    def __init__(self, rnn, seq_len, abstraction_type=None):
        """"""
        self.rnn = rnn
        self.seq_len = seq_len
        if abstraction_type is None:
            if rnn.get_input_size() > rnn.get_hidden_size():
                self.in_abstraction_type = INPUT_ON_START_ONE_OUTPUT
            else:
                self.in_abstraction_type = INPUT_ON_DEMAND_ONE_OUTPUT
        else:
            self.abstraction_type = abstraction_type

    def build_abstraction(self):
        if self.abstraction_type == INPUT_ON_START_ONE_OUTPUT:
            return self._build_ios()
        elif self.abstraction_type == INPUT_ON_START_OUTPUT_ON_START:
            return self._build_ios_oos()
        elif self.abstraction_type == INPUT_ON_START_OUTPUT_ON_DEMAND:
            return self._build_ios_ood()
        elif self.abstraction_type == INPUT_ON_DEMAND_ONE_OUTPUT:
            return self._build_iod()
        elif self.abstraction_type == INPUT_ON_DEMAND_OUTPUT_ON_START:
            return self._build_iod_oos()
        elif self.abstraction_type == INPUT_ON_DEMAND_OUTPUT_ON_DEMAND:
            return self._build_iod_ood()
        else:
            return None

    def _build_ios(self):
        input_hidden = self._build_ios_first_layer()
        hidden_layers = []
        for l in range(1, self.seq_len):
            hidden_layers.append(self._build_ios_hidden_layer_at_step(l))
        hidden_output = self.rnn.get_hidden_output()
        if hidden_layers:
            for l in range(len(hidden_layers)):
                if l > 0:
                    _, prev_cols = hidden_layers[l - 1].shape
                    rows, cols = hidden_layers[l].shape
                    if prev_cols != rows:
                        difference = abs(prev_cols - rows)
                        hidden_padding = np.zeros((difference, cols))
                        hidden_layers[l] = np.concatenate((hidden_layers[l], hidden_padding))

            padding = np.zeros((hidden_layers[-1].shape[1] - hidden_output.shape[0], hidden_output.shape[1]))
            hidden_output = np.concatenate((hidden_output, padding))
        return FFNN(input_hidden, hidden_layers, hidden_output)

    def _build_ios_oos(self):
        input_hidden = self._build_ios_first_layer()
        rnn_hidden_size = self.rnn.get_hidden_size()
        first_hidden = self._build_ios_hidden_layer_at_step(1)
        first_output = np.block([
                [self.rnn.get_hidden_output(),
                 self.rnn.get_hidden_hidden(),
                 np.zeros((rnn_hidden_size,
                          (self.seq_len - 2) * rnn_hidden_size))],
                [np.kron(np.eye(self.seq_len - 2), np.eye(rnn_hidden_size))]
            ])
        hidden_layers = [first_hidden, first_output]
        for l in range(3, self.seq_len):
            hidden_layer = np.block([
                [np.eye((l-2)*self.rnn.get_output_size()),
                 self.rnn.get_hidden_output(),
                 self.rnn.get_hidden_hidden(),
                 np.zeros((rnn_hidden_size,
                          (self.seq_len - l) * rnn_hidden_size))],
                [np.kron(np.eye(self.seq_len - l), np.eye(rnn_hidden_size))]
            ])
            hidden_layers.append(hidden_layer)
        output_layer = np.block([
          np.eye(self.seq_len - 1) * self.rnn.get_output_size(),
          self.rnn.get_hidden_output()])
        return FFNN(input_hidden, hidden_layers, output_layer)

    def _build_ios_ood(self):
        input_hidden = self._build_ios_first_layer()
        rnn_hidden_size = self.rnn.get_hidden_size()
        first_hidden = np.block([
            [np.eye(rnn_hidden_size),
             self.rnn.get_hidden_hidden(),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_hidden_size))],
            [np.zeros((rnn_hidden_size, rnn_hidden_size)),
             np.eye(rnn_hidden_size),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_hidden_size))],
            [np.zeros((rnn_hidden_size * (self.seq_len - 2), 2 * rnn_hidden_size)),
             np.eye(rnn_hidden_size * (self.seq_len - 2))]
        ])
        hidden_layers = [first_hidden]
        for l in range(2, self.seq_len):
            hidden_layer = np.block([
              [np.eye((l - 1) * rnn_hidden_size),
               np.zeros(((l - 1) * rnn_hidden_size,
                         (self.seq_len - l) * rnn_hidden_size))],
              [np.zeros((rnn_hidden_size,
                         (l - 1) * rnn_hidden_size)),
               np.eye(rnn_hidden_size),
               np.zeros((rnn_hidden_size,
                         (self.seq_len - l - 1) * rnn_hidden_size))],
              [np.zeros((rnn_hidden_size * (self.seq_len - l - 1),
                         l * rnn_hidden_size)),
               np.eye(rnn_hidden_size*(self.seq_len - l - 1))]
            ])
            hidden_layers.append(hidden_layer)
        output_layer = np.kron(self.rnn.get_hidden_output, np.eye(self.seq_len))
        return FFNN(input_hidden, hidden_layers, output_layer)

    def _build_ios_hidden_layer_at_step(self, l):
        rnn_hidden_size = self.rnn.get_hidden_size()
        hidden_hidden = self.rnn.get_hidden_hidden()
        first_block_zeroes = np.zeros((rnn_hidden_size, (self.seq_len - l) * rnn_hidden_size))
        first_block = [hidden_hidden, first_block_zeroes]

        kron = np.kron(np.eye(self.seq_len - l), np.eye(rnn_hidden_size))

        leftover = np.zeros((kron.shape[0], hidden_hidden.shape[1] + first_block_zeroes.shape[1] - kron.shape[1]))
        second_block = [kron, leftover]
        return np.block([first_block, second_block])

    def _build_iod(self):
        input_hidden = self._construct_unrolled_input_hidden()
        hidden_layers = []
        for i in range(2, self.seq_len + 1):
            hidden_layers.append(
                self._construct_unrolled_hidden_at_step(i))
        return FFNN(input_hidden, hidden_layers, self.rnn.get_hidden_output())

    def _build_iod_oos(self):
        input_hidden = self._construct_unrolled_input_hidden()
        rnn_hidden_size = self.rnn.get_hidden_size()
        rnn_input_size = self.rnn.get_input_size()
        first_hidden = np.block([
            [self.rnn.get_hidden_output(),
             self.rnn.get_hidden_hidden(),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_input_size))],
            [np.zeros((rnn_hidden_size, self.rnn.get_output_size())),
             self.rnn.get_input_hidden(),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_input_size))],
            [np.zeros(((self.seq_len - 2) * rnn_hidden_size,
                       self.rnn.get_output_size() + rnn_hidden_size)),
             np.eye((self.seq_len - 2) * rnn_input_size)]
        ])
        hidden_layers = [first_hidden]
        rnn_output_size = self.rnn.get_output_size()
        for i in range (1, self.seq_len - 1):
            hidden_layer = np.block([
                [np.eye(i * rnn_output_size),
                 np.zeros((i * rnn_output_size,
                           rnn_output_size +
                           rnn_hidden_size +
                           (self.seq_len - i - 2) * rnn_input_size))],
                [np.zeros(rnn_hidden_size, i * rnn_output_size),
                 self.rnn.get_hidden_output(),
                 self.rnn.get_hidden_hidden(),
                 np.zeros((rnn_hidden_size,
                          (self.seq_len - i - 2) * rnn_input_size))],
                [np.zeros((rnn_input_size, (i + 1) * rnn_output_size)),
                 self.rnn.get_input_hidden(),
                 np.zeros((rnn_input_size,
                           (self.seq_len - i - 2) * rnn_input_size))],
                [np.zeros(((self.seq_len - i - 2) * rnn_input_size,
                           (i + 1) * rnn_output_size + rnn_hidden_size)),
                 np.eye((self.seq_len - i - 2) * rnn_input_size)]
            ])
            hidden_layers.append(hidden_layer)
        output_layer = np.block([
            [np.eye((self.seq_len - 1) * rnn_output_size),
             np.zeros(((self.seq_len - 1) * rnn_output_size, rnn_output_size))],
            [np.zeros((rnn_hidden_size, (self.seq_len - 1) * rnn_output_size)),
             self.rnn.get_hidden_output()]
        ])
        return FFNN(input_hidden, hidden_layers, output_layer)

    def _build_iod_ood(self):
        input_hidden = self._construct_unrolled_input_hidden()
        rnn_output_size = self.rnn.get_output_size()
        rnn_hidden_size = self.rnn.get_hidden_size()
        rnn_input_size = self.rnn.get_input_size()
        first_hidden = np.block([
            [np.eye(rnn_hidden_size),
             self.rnn.get_hidden_hidden(),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_input_size))],
            [np.zeros((rnn_input_size, rnn_hidden_size)),
             self.rnn.get_input_hidden(),
             np.zeros((rnn_hidden_size, (self.seq_len - 2) * rnn_input_size))],
            [np.zeros(((self.seq_len - 2) * rnn_input_size),
                      2 * rnn_hidden_size),
             np.eye((self.seq_len - 2) * rnn_input_size)]
        ])
        hidden_layers = [first_hidden]
        for i in range(1, self.seq_len - 2):
            hidden_layer = np.block([
                [np.eye(i * rnn_hidden_size),
                 np.zeros((i * rnn_hidden_size,
                           2 * hidden_layer +
                           (self.seq_len - i - 2) * rnn_input_size))] ,
                [np.zeros((rnn_hidden_size, i * rnn_hidden_size)),
                 np.eye(rnn_hidden_size),
                 self.rnn.get_hidden_hidden(),
                 np.zeros((rnn_hidden_size,
                           (self.seq_len - i - 2) * rnn_input_size))],
                [np.zeros((rnn_input_size, (i + 1) * rnn_hidden_size)),
                 self.rnn.get_input_hidden(),
                 np.zeros((rnn_input_size,
                           (self.seq_len - i - 2) * rnn_input_size))],
                [np.zeros(((self.seq_len - i - 2) * rnn_input_size,
                           (i + 2) * rnn_hidden_size)),
                 np.eye((self.seq_len - i - 2) * rnn_input_size)]
            ])
            hidden_layers.append(hidden_layer)
        last_hidden = np.block([
            [np.eye((self.seq_len - 2) * rnn_hidden_size),
             np.zeros(((self.seq_len - 2) * rnn_hidden_size,
                       2 * rnn_hidden_size))],
            [np.zeros(rnn_hidden_size, (self.seq_len - 2) * rnn_hidden_size),
             np.eye(rnn_hidden_size),
             self.rnn.get_hidden_hidden()],
            [np.zeros((rnn_input_size, (self.seq_len - 1) * rnn_hidden_size)),
             self.rnn.get_input_hidden()]
        ])
        hidden_layers.append(last_hidden)
        output_layer = np.kron(self.rnn.get_hidden_output(),
                               np.eye(self.seq_len))
        return FFNN(input_hidden, hidden_layers, output_layer)

    def _construct_unrolled_input_hidden(self):
        len_of_future_sequences = self.rnn.get_input_size() * (self.seq_len - 1)

        future_steps_to_hidden = np.zeros((len_of_future_sequences,
                                           self.rnn.get_hidden_size()))
        first_step_to_copies = np.zeros((self.rnn.get_input_size(),
                                         len_of_future_sequences))
        copies_of_future = np.eye(len_of_future_sequences)

        return np.block([
            [self.rnn.get_input_hidden(), first_step_to_copies],
            [future_steps_to_hidden, copies_of_future]
        ])

    def _construct_unrolled_hidden_at_step(self, step):
        len_of_future_sequences = self.rnn.get_input_size() * (
          self.seq_len - step)

        future_steps_to_hidden = np.zeros((len_of_future_sequences,
                                           self.rnn.get_hidden_size()))
        output_to_copies = np.zeros((self.rnn.get_hidden_size(),
                                     len_of_future_sequences))
        current_step_to_copies = np.zeros((self.rnn.get_input_size(),
                                           len_of_future_sequences))
        copies_of_future = np.eye(len_of_future_sequences)

        return np.block([
            [self.rnn.get_hidden_hidden(), output_to_copies],
            [self.rnn.get_input_hidden(), current_step_to_copies],
            [future_steps_to_hidden, copies_of_future]
        ])

    def _build_ios_first_layer(self):
        return np.kron(np.eye(self.seq_len), self.rnn.get_input_hidden())


def plot_abstraction(rnn_model, abstraction_type, seq_len, plot_name):
    rnn = keras_rnn_checker.construct_rnn_from_keras_rnn(rnn_model)
    abstractor = RNNAbstractor(rnn, seq_len, abstraction_type)
    abstraction = abstractor.build_abstraction()
    ffnn = keras_rnn_checker.construct_keras_ffnn_from_rnn(abstraction, seq_len)
    keras.utils.plot_model(ffnn, to_file=plot_name, show_shapes=True)
    return ffnn, abstraction
