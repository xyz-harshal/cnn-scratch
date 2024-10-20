import numpy as np
from tqdm import tqdm

class Pooling:
    def __init__(self,input_shape,kernel_shape):
        self.input_shape=input_shape
        self.kernel_shape=kernel_shape

    def pool(self,A_prev,stride=1):
        m,h,w,c=self.input_shape
        k_h,k_w=self.kernel_shape

        n_h=int((h-k_h)/stride)+1
        n_w=int((w-k_w)/stride)+1
        A_pool=np.zeros((m,n_h,n_w,c))

        for i in tqdm(range(n_h),desc="Pooling Progress"):
            for j in range(n_w):
                vert_start=stride*i
                vert_end=vert_start+k_h
                horz_start=stride*j
                horz_end=horz_start+k_w

                A_slice=A_prev[:,vert_start:vert_end,horz_start:horz_end,:]
                A_pool[:,i,j,:]=np.max(A_slice,axis=(1,2))
        return A_pool
