import numpy as np
import pandas as pd
from src.layers import ConvLayer,Pooling

data=pd.read_csv('./data/mnist_train.csv')
data=np.array(data)

X=data[:10000,1:]
Y=data[:10000,0]
X=X/255
m,n=X.shape
X=X.reshape(m,28,28,1)

reluActivation=lambda A:np.maximum(0,A)

conv_layer1=ConvLayer(X.shape,(15,15,16))
Z1=conv_layer1.forward(X,pad=0,stride=1)
A1=reluActivation(Z1)

pool_layer1=Pooling(A1.shape,(4,4))
P1=pool_layer1.pool(A1,stride=1)

conv_layer2=ConvLayer(P1.shape,(5,5,32))
Z2=conv_layer2.forward(P1,pad=0,stride=1)
A2=reluActivation(Z2)

pool_layer2=Pooling(A2.shape,(4,4))
P2=pool_layer2.pool(A2,stride=1)
