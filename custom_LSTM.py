import numpy as np
import torch
import torch.nn as nn

import syft as sy
from syft.execution.plan import Plan



class LSTMCell(nn.Module):
    """
    Python implementation of LSTMCell for MPC
    This class overrides the torch.nn.LSTMCell
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(LSTMCell, self).__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity


        # Input Gate
        self.fc_xi = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Forget Gate
        self.fc_xf = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hf = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Cell Gate
        self.fc_xc = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hc = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Output Gate
        self.fc_xo = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_ho = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.init_parameters()
        
    def init_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        
#     def init_hidden(self, input):
#         """
#         TODO: Not being used 
#         This method initializes a hidden state when no hidden state is provided
#         in the forward method. It creates a hidden state with zero values.
#         """
# #         h = torch.zeros(input.shape[0], self.hidden_size, dtype=input.dtype, device=input.device)
#         h = torch.zeros(input.shape[0], self.hidden_size)
#         if input.has_child() and isinstance(input.child, PointerTensor):
#             h = h.send(input.child.location)
#         if input.has_child() and isinstance(input.child, precision.FixedPrecisionTensor):
#             h = h.fix_precision()
#             child = input.child
#             if isinstance(child.child, AdditiveSharingTensor):
#                 crypto_provider = child.child.crypto_provider
#                 owners = child.child.locations
#                 h = h.share(*owners, crypto_provider=crypto_provider)
#         return h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x, hc=None):

        if hc is None:
            batch_size = x.shape[1]
            hc = (self.init_hidden(batch_size), self.init_hidden(batch_size))
        h, c = hc
        
#         print('LSTMCell', type(x), x.shape)
#         print('Hidden', h, h.shape)
#         print('C t-1', c, c.shape)
        x_i = self.fc_xi(x)
        h_i = self.fc_hi(h)
        x_f = self.fc_xf(x)
        h_f = self.fc_hf(h)
        x_c = self.fc_xc(x)
        h_c = self.fc_hc(h)
        x_o = self.fc_xo(x)
        h_o = self.fc_ho(h)
        
        inputgate = (x_i + h_i).sigmoid()
        forgetgate = (x_f + h_f).sigmoid()
        cellgate = (x_c + h_c).tanh()
        outputgate = (x_o + h_o).sigmoid()

#         c_ = torch.mul(forgetgate, c) + torch.mul(inputgate, cellgate)
        c_ = (forgetgate * c) + (inputgate * cellgate)

#         h_ = torch.mul(outputgate, torch.tanh(c_))
        h_ = outputgate * c_.tanh()

        return h_, c_


class LSTM(nn.Module):
    """
    V2
    Python implementation of LSTM for MPC
    This class overrides the torch.nn.LSTM
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        nonlinearity=None,
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1
#         self.is_lstm = base_cell is LSTMCell
        self.nonlinearity = nonlinearity
    
        # Dropout layers
        # TODO: implement a nn.Dropout class for PySyft
        # Link to issue: https://github.com/OpenMined/PySyft/issues/2500

        # Build RNN forward layers
        sizes = [input_size, *(hidden_size for _ in range(self.num_layers - 1))]
#         print('Sizes', sizes)
        self.rnn_forward = nn.ModuleList(
            (LSTMCell(sz, hidden_size, bias, nonlinearity) for sz in sizes)
        )
        
        self.lstm_cell = LSTMCell(self.input_size, self.hidden_size, self.bias, self.nonlinearity)

#         # Build RNN backward layers, if needed
#         if self.bidirectional:
#             self.rnn_backward = nn.ModuleList(
#                 (base_cell(sz, hidden_size, bias, nonlinearity) for sz in sizes)
#             )

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x, hc=None):
        
        batch_size = x.shape[1]
        seq_len = x.shape[0]
        
#         print('x', x.shape)
#         print('batch size', batch_size)
#         print('seq_len', seq_len)
        
        if hc is None:
            print('Init hc...')
            hc = (self.init_hidden(batch_size), self.init_hidden(batch_size))
            
        # Run through rnn in the forward direction
#         output = x.new(seq_len, batch_size, self.hidden_size).zero_()

#         hc = torch.stack([*hc])  # TODO: stack does not work?
        for t in range(seq_len):
#             hc_next = torch.zeros_like(hc)
#             print('hc_next', hc_next.shape)
#             for layer in range(self.num_layers):
#                 print('layer', layer)
# #                 input_ = x[t, :, :] if layer == 0 else hc_next[0][layer - 1, :, :].clone()
#                 input_ = x[t, :] if layer == 0 else hc_next[0][layer - 1].clone()
# #                 print('input', input_)
# #                 hc_next[:, layer, :, :] = torch.stack(self.rnn_forward[layer](input_, hc[:, layer, :, :]))
#                 hc_next[:, layer] = torch.stack(self.rnn_forward[layer](input_, hc[:, layer]))
    
            input_ = x.select(0, t).view(1, -1)
        
#             if t == seq_len - 1:
#                 print('input_', input_, input_.shape)

#             input_ = x[t, :]
            hc = self.lstm_cell(input_, hc)
    
#             if t == seq_len - 1:
#                 print('hc', hc, hc[0].shape, hc[1].shape)
                
#             output[t, :, :] = hc_next[0][-1, :, :]
#             output = hc_next[0][-1]
                
#         return output, tuple(hc_next)
        return hc
