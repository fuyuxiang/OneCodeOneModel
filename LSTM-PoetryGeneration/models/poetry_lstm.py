import torch.nn as nn


class PoetryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(PoetryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM层
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # 嵌入层
        embedded = self.embedding(x)

        # LSTM层
        lstm_out, hc = self.lstm(embedded)

        # 全连接层
        output = self.fc(lstm_out)
        return output, hc
