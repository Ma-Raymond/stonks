import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
# Get the current date

current_date = datetime.now()

# Format the date as "YYYY-MM-DD"
formatted_date = current_date.strftime("%Y-%m-%d")

# data = yf.download(tickers='VFV',start='2012-03-11',end='2023-07-01')
data = yf.download(tickers='VFV.TO',start='2012-03-11')
# data = yf.download(tickers='VFV',start='2012-03-11',end=f'{formatted_date}',period = "5d")

# Technical Indicators
data['RSI'] = ta.rsi(data.Close, length=15)
data['EMAF'] = ta.ema(data.Close, length=20)
data['EMAM'] = ta.ema(data.Close, length=100)
data['EMAS'] = ta.ema(data.Close, length=150)

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace=True)
data.drop(['Volume','Close','Date'],axis=1,inplace=True)
data_set = data.iloc[:,0:11]
pd.set_option('display.max_columns',None)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
sc = MinMaxScaler(feature_range=(0,1))
data_scaled = sc.fit_transform(data_set)


x =[]

backcandles = 20 # # of days to look back?
print(data_scaled.shape)
for j in range(8):
    x.append([])
    for i in range(backcandles,data_scaled.shape[0]):
        x[j].append(data_scaled[i-backcandles:i,j])
        
# [    ]
# [[]  ]
# [[len(10)],[len(10)],[len(10)],[len(10)],[len(10)],[len(10)],[len(10)],[len(10)]]
# [[len(10)*2695]*8]
# [[[10]*2695]*8]

print(np.array(x).shape)

x = np.moveaxis(x,[0],[2])
x,yi = np.array(x), np.array(data_scaled[backcandles:,-1])
y = np.reshape(yi,(len(yi),1))
print(y.shape)


splitlimit = int(len(x)*0.8)
x_train,x_test = x[:splitlimit],x[splitlimit:]
y_train,y_test = y[:splitlimit],y[splitlimit:]

import torch
import torch.nn as nn
from torch.autograd import Variable 
x_train_tensors = Variable(torch.Tensor(x_train))
x_test_tensors = Variable(torch.Tensor(x_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

class LSTM1(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,seq_length):
        super(LSTM1,self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 = nn.Linear(hidden_size,128)
        self.fc = nn.Linear(128,num_classes)
        
        self.relu = nn.ReLU()
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        output,(hn,cn) = self.lstm(x,(h_0,c_0))
        hn = hn.view(-1,self.hidden_size) # It turns it into whatever but needs to be (something,hidden_state)

        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    

num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 8 #number of features
hidden_size = 20 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train_tensors.shape[1]) #our lstm class 
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

for epoch in range(num_epochs):
  outputs = lstm1.forward(x_train_tensors) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
    

x_test_1 = Variable(torch.Tensor(x_test))
print(x_test_1.shape)
y_pred = lstm1(x_test_1)
y_pred = y_pred.data.numpy()

plt.figure(figsize=(16,8))
plt.plot(y_test,color='black',label='Test')
plt.plot(y_pred,color='green',label='Pred')
plt.legend()
plt.show()