import numpy as np
from rns_verify.ffnn import FFNN


class RNN(object):
    """

    Attributes:
        input_hidden:
        hidden_hidden:
        hidden_output:
    """

    def __init__(self, input_hidden, hidden_hidden, hidden_output, hidden_bias):
        """"""
        self.input_hidden = input_hidden
        self.hidden_hidden = hidden_hidden
        self.hidden_output = hidden_output
        self.hidden_bias = hidden_bias

    def get_input_size(self):
        return self.input_hidden.shape[0]

    def get_hidden_size(self):
        return self.hidden_hidden.shape[0]

    def get_output_size(self):
        return self.hidden_output.shape[1]

    def get_input_hidden(self):
        return self.input_hidden

    def get_hidden_hidden(self):
        return self.hidden_hidden

    # def get_hidden_bias(self):
    #     return self.hidden_bias
    #
    def get_hidden_output(self):
        return self.hidden_output

    def unroll(self, no_of_sequences=1):
        input_size = self.get_input_size()
        hidden_size = self.get_hidden_size()
        unrolled_input_hidden = self.construct_unrolled_input_hidden(no_of_sequences, input_size, hidden_size)
        unrolled_hidden_layers = []
        for i in range(2, no_of_sequences + 1):
            unrolled_hidden_layers.append(
                self.construct_unrolled_hidden_at_step(i, no_of_sequences, input_size, hidden_size))
        return FFNN(unrolled_input_hidden, unrolled_hidden_layers, self.hidden_output)

    def construct_unrolled_input_hidden(self, no_of_sequences, input_size, hidden_size):
        length_of_future_sequences = input_size * (no_of_sequences - 1)

        future_steps_to_hidden = np.zeros((length_of_future_sequences, hidden_size))
        first_step_to_copies = np.zeros((input_size, length_of_future_sequences))
        copies_of_future = np.eye(length_of_future_sequences)

        return np.block([
            [self.input_hidden, first_step_to_copies],
            [future_steps_to_hidden, copies_of_future]
        ])

    def construct_unrolled_hidden_at_step(self, step, no_of_sequences, input_size, hidden_size):
        length_of_future_sequences = input_size * (no_of_sequences - step)

        future_steps_to_hidden = np.zeros((length_of_future_sequences, hidden_size))
        output_to_copies = np.zeros((hidden_size, length_of_future_sequences))
        current_step_to_copies = np.zeros((input_size, length_of_future_sequences))
        copies_of_future = np.eye(length_of_future_sequences)

        return np.block([
            [self.hidden_hidden, output_to_copies],
            [self.input_hidden, current_step_to_copies],
            [future_steps_to_hidden, copies_of_future]
        ])
