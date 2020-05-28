import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, word_dim, max_len, class_num):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(max_len * word_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_num), 
        )

    def forward(self, x):
        out = self.mlp(x.view(x.size(0), -1))
        return out
    

class CNN(nn.Module):
    def __init__(self, word_dim, max_len, class_num):
        super(CNN, self).__init__()
        window_sizes = [2, 3, 4]
        # window_sizes = [3, 4, 5, 6]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=word_dim, out_channels=128, kernel_size=h),
                nn.BatchNorm1d(num_features=128), 
                # nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=max_len - h + 1)
            )
            for h in window_sizes
        ])
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * len(window_sizes), out_features=64),
            nn.BatchNorm1d(num_features=64), 
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=class_num),
        )
            

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1)) 
        out = self.fc(out)
        return out


class LSTM(nn.Module):
    def __init__(self, word_dim, class_num, device):
        super(LSTM, self).__init__()
        self.device = device
        self.rnn = nn.LSTM(input_size=word_dim, hidden_size=256, num_layers=1, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(num_features=128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, class_num),
        )


    def attention(self, output, state):
        h = state.view(-1, 256 * 2, 1)
        weights = torch.bmm(output, h).squeeze(2)
        weights = F.softmax(weights, 1)
        context = torch.bmm(output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        return context


    def forward(self, x):
        
        input = x.permute(1, 0, 2)
        h0 = Variable(torch.zeros(1 * 2, len(x), 256).to(self.device))
        c0 = Variable(torch.zeros(1 * 2, len(x), 256).to(self.device))

        output, (hn, cn) = self.rnn(input, (h0, c0))
        output = output.permute(1, 0, 2) 

        output = self.attention(output, hn)

        return self.fc(output)


class GRU(nn.Module):
    def __init__(self, word_dim, class_num, device):
        super(GRU, self).__init__()
        self.device = device
        self.rnn = nn.GRU(input_size=word_dim, hidden_size=256, num_layers=1, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.BatchNorm1d(num_features=128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, class_num),
        )


    def attention(self, output, state):
        h = state.view(-1, 256 * 2, 1) 
        weights = torch.bmm(output, h).squeeze(2)
        weights = F.softmax(weights, 1)
        context = torch.bmm(output.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        return context


    def forward(self, x):
        
        input = x.permute(1, 0, 2)
        h0 = Variable(torch.zeros(1 * 2, len(x), 256).to(self.device)) 

        output, hn = self.rnn(input, h0)
        output = output.permute(1, 0, 2) 

        output = self.attention(output, hn)

        return self.fc(output)
