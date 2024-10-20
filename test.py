import numpy as np
name=np.random.randn(10,10,5,3)
lame=name.shape
print(lame,type(lame))

def convo(A_prev,K,stride=1,pad=0):
    m,h,w,c=A_prev.shape
    kernels=np.random.randn(K[0],K[1],c,K[2])*0.01
    k_h,k_w,k_d,k_c=kernels.shape
    bias=np.zeros((1,1,1,k_c))
    n_h=int((h-k_h+2*pad)/stride)+1
    n_w=int((w-k_w+2*pad)/stride)+1
    n_c=k_c
    A=np.zeros((m,n_h,n_w,n_c))
    for n in range(m):
        a_prev=A_prev[n]
        print(n," convolued")
        for n_h_crop in range(n_h):
            vert_start=stride*n_h_crop
            vert_end=vert_start+k_h
            for n_w_crop in range(n_w):
                horz_start=stride*n_w_crop
                horz_end=horz_start+k_w
                for C in range(n_c):
                    a_prev_crop=a_prev[vert_start:vert_end,horz_start:horz_end,:]
                    k=kernels[:,:,:,C]
                    b=bias[:,:,:,C]
                    A[n,n_h_crop,n_w_crop,C]=cross_correlation(a_prev_crop,k,b)
    return A


def pool(A_prev,P,stride=1):
    m,h,w,c=A_prev.shape
    p_h,p_w,p_c=P[0],P[1],c
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
