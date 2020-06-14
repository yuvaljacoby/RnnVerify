class FFNN(object):
    """
    Attributes:
        input_hidden:
        hidden_layers:
        hidden_output:
    """

    def __init__(self, input_hidden, hidden_layers, hidden_output):
        """"""
        self.input_hidden = input_hidden
        self.hidden_layers = hidden_layers
        self.hidden_output = hidden_output

    def get_input_hidden(self):
        return self.input_hidden

    def get_hidden_layers(self):
        return self.hidden_layers

    def get_hidden_output(self):
        return self.hidden_output

    def get_layers(self):
        layers = [self.input_hidden]
        for layer in self.hidden_layers:
            layers.append(layer)
        layers.append(self.hidden_output)
        return layers



