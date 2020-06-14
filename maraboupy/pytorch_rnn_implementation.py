import torch
import torch.nn as nn
import numpy as np

in_dim = 3
hidden_dim = 2
out_dim = 2
timesteps = 2
# torch.manual_seed(10)

def ReLU(x):
    return max(x, 0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_step, activation):
        # the num_layers here is the number of RNN units
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = input_size,hidden_size=hidden_size,
                          num_layers = 1,batch_first=True, nonlinearity = activation)
        self.out = nn.Linear(hidden_size , output_size )
        self.hidden_r = None

    def forward(self,X):
        # X is of size (batch, seq_len, input_size)
        # print(self.a_0)
        r_out, h_n = self.rnn(X, self.hidden_r) # RNN usage: output, hn = rnn(input,h0)
        self.hidden_r = r_out
        # print("r_out:", r_out)
        # print('h_n:', h_n)
        out = self.out(r_out[:,-1,:])
        return out      


if __name__ == "__main__":
    for i in range(300):
        print(i)
        torch.manual_seed(i)
        small_model = RNN(in_dim, hidden_dim,out_dim,timesteps,'relu')
        small_model.eval()

        x1 = torch.from_numpy(np.array([[1,1,1]])[None,:].astype(np.float32))

        state = small_model.state_dict()

        wi = state['rnn.weight_ih_l0']
        bi = state['rnn.bias_ih_l0']
        wh = state['rnn.weight_hh_l0']
        # print(wh.shape)
        bh = state['rnn.bias_hh_l0']
        rnn_bias = bi + bh
        wo = state['out.weight']
        bo = state['out.bias']
        # print("bi", bi)
        # print("bh", bh)
        # print("bo", bo)

        R01 = ReLU(x1[0,0,0] * wi[0,0] + x1[0,0,1] * wi[0,1] + x1[0,0,2] * wi[0,2] + rnn_bias[0])
        R02 = ReLU(x1[0,0,0] * wi[1,0] + x1[0,0,1] * wi[1,1] + x1[0,0,2] * wi[1,2] + rnn_bias[1])

        # print("R01", R01)
        # print("R02", R02)
        o1 = R01 * wo[0,0] + R02 * wo[0,1] + bo[0]
        o2 = R01 * wo[1,0] + R02 * wo[1,1] + bo[1]
        actual = small_model(x1)[0]
        assert torch.allclose(actual[0], o1), i 
        assert torch.allclose(actual[1], o2), i 
        # print(small_model(x1)[0][0], o1)
        # print(small_model(x1)[0][1], o2)

        # x2 = torch.from_numpy(np.array([[0.3,1.3,1.2]])[None,:].astype(np.float32))
        # print("\nstarting second input!\n")
        R11 = ReLU(x1[0,0,0] * wi[0,0] + x1[0,0,1] * wi[0,1] + x1[0,0,2] * wi[0,2] +
                R01 * wh[0,0] + R02 * wh[0,1] + 
                rnn_bias[0])
        R12 = ReLU(x1[0,0,0] * wi[1,0] + x1[0,0,1] * wi[1,1] + x1[0,0,2] * wi[1,2] +
                R01 * wh[1,0] + R02 * wh[1,1] + 
                rnn_bias[1])
        # print("R11", R11)
        # print("R12", R12)
        o11 = R11 * wo[0,0] + R12 * wo[0,1] + bo[0]
        o12 = R11 * wo[1,0] + R12 * wo[1,1] + bo[1]
        actual = small_model(x1)[0]
        # print('o11', o11)
        # print('actual[0]', actual[0])
        # print(actual[0] - o11)
        assert torch.allclose(actual[0], o11), i 
        assert torch.allclose(actual[1], o12), i 

