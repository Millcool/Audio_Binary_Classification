from config import *


class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, input_dim=80, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.clf = nn.Linear(hidden_size, 2)
        self._fc = torch.nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=2, bias=True))

    def forward(self, input, length=CFG.width - 1):
        # input: (batch_size, hidden_size, seq_len)
        out, _ = self.lstm(input.transpose(-1, -2))
        out = out[:, CFG.width - 1, :]
        # output: (batch_size, seq_len, hidden_size)

        #         last_hidden = torch.gather(
        #             output,
        #             dim =1,
        #             index = length.sub(1).view(-1, 1, 1).expand(-1, -1, self.hidden_size)
        #         )
        # logits = self.clf(last_hidden.squeeze(dim=1))
        # print(f'OUT SHAPE: {out.shape}')
        out = self._fc(out)
        return out
