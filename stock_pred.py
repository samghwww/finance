
import pandas as pd
import numpy as np

import torch


from torch import nn
from plotly import graph_objects as go
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out)

        return out

model = torch.load('./models/model.pth')

dframe = pd.read_csv("data/gzmt.csv")

# Create traces
layout = go.Layout(
    title='A股票价格走势',
    xaxis=dict(title='日期'),
    yaxis=dict(title='价格'),
)
x = dframe.loc[:,'date'].values
y = dframe.loc[:,'close_price'].values
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Real"))
#fig.show()

df = dframe.loc[:,'open_price':'highest_price']
scaler = MinMaxScaler(feature_range=(-1, 1))
scl = scaler.fit_transform(df.values)

BATCH_SIZE = 20
feature_size = df.columns.size

dlen = len(scl) - (len(scl) % BATCH_SIZE)

data = torch.from_numpy(scl[:dlen].astype(np.float32))
x_to_pred = data.view(-1, BATCH_SIZE, feature_size)
y_pred = model(x_to_pred)

pred_value = y_pred.detach().numpy().reshape(-1, 1)

close_price = y.reshape(-1, 1)
scaler_for_close_price = MinMaxScaler(feature_range=(-1, 1))
close_price_scale = scaler_for_close_price.fit_transform(close_price)

pred_value = scaler_for_close_price.inverse_transform(pred_value).reshape(-1)
true_value = y[:dlen]

# Create traces
layout = go.Layout(
    title='A股票价格走势',
    xaxis=dict(title='日期'),
    yaxis=dict(title='价格'),
)
x = dframe.iloc[:,1].values[:dlen]
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=x, y=true_value, mode="lines", name="Real"))
fig.add_trace(go.Scatter(x=x, y=pred_value, mode="lines", name="Predict")) # marks
fig.show()
