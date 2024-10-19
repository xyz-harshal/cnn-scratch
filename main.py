import math
import numpy as np
import pandas as pd

data=pd.read_csv('./mnist_train.csv')

data=np.array(data)

X=data[:1000,1:]
Y=data[:1000,0]
X=X/255
m,n=X.shape
X=X.reshape(m,28,28)

def cross_correlation(a_part,kernel,bias):
    a=a_part*kernel
    z=np.sum(a)+float(bias)
    return z

def convo1(A_prev,pad,stride):
    m,h,w=A_prev.shape
    kernels=np.random.randn(15,15,16)
    h_k,w_k,c_k=kernels.shape
    bias=np.zeros((c_k))
    n_h=int((h-h_k+2*pad)/stride +1)
    n_w=int((w-w_k+2*pad)/stride +1)
    n_c=c_k
    A=np.zeros((m,n_h,n_w,n_c))
    for n in range(m):
        x_crop=X[n]
        print(n," convolued")
        for n_h_crop in range(n_h):
            vert_start=stride*n_h_crop
            vert_end=vert_start+h_k
            for n_w_crop in range(n_w):
                horz_start=stride*n_w_crop
                horz_end=horz_start+w_k
                for c in range(n_c):
                    a_corr=x_crop[horz_start:horz_end,vert_start:vert_end]
                    w=kernels[:,:,c]
                    b=bias[c]
                    A[n,n_h_crop,n_w_crop,c]=cross_correlation(a_corr,w,b)
    return A


def reluActivation(A):
    return np.maximum(0,A)

def getMax(a_crop):
    return np.max(a_crop)

def pool1(A_prev,stride=1):
    m,h,w,c=A_prev.shape
    p_h,p_w,p_c=4,4,c
    n_h=int((h-p_h)/stride)+1
    n_w=int((w-p_w)/stride)+1
    A_pool=np.zeros((m,n_h,n_w,c))
    for n in range(m):
        A_prev_crop=A_prev[n]
        print(n," pooled")
        for n_h_crop in range(n_h):
            vert_start=stride*n_h_crop
            vert_end=vert_start+p_h
            for n_w_crop in range(n_w):
                horz_start=stride*n_w_crop
                horz_end=horz_start+p_w
                for C in range(c):
                    a_prev_crop=A_prev_crop[vert_start:vert_end,horz_start:horz_end,:]
                    A_pool[n,n_h_crop,n_w_crop,C]=getMax(a_prev_crop[:,:,C])
    return A_pool


Z=convo1(X,pad=0,stride=1)
A=reluActivation(Z)
A_P=pool1(A,stride=1)
print(A_P.shape)
print(A_P)
