from tqdm import tqdm
import numpy as np

class ConvLayer:
    def __init__(self,input_shape,kernel_shape):
        self.input_shape=input_shape
        self.kernel_shape=kernel_shape
        #parameters initializations
        self.kernels=np.random.randn(self.kernel_shape[0],self.kernel_shape[1],self.input_shape[3],self.kernel_shape[2])*0.01
        self.bias=np.zeros((1,1,1,self.kernel_shape[2]))

    def forward(self,A_prev,stride=1,pad=0):
        m,h,w,d=self.input_shape
        k_h,k_w,k_d,k_c=self.kernels.shape

        n_h=int((h-k_h+2*pad)/stride)+1
        n_w=int((w-k_w+2*pad)/stride)+1
        #padding is done here:
        if pad>0:A_prev_pad=np.pad(A_prev,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant')
        else: A_prev_pad=A_prev
        #initializing the output
        Z=np.zeros((m,n_h,n_w,k_c))
        #vectorize solution
        for i in tqdm(range(n_h),desc="Convolution Progress"):
            for j in range(n_w):
                vert_start=i*stride
                vert_end=vert_start+k_h
                horz_start=j*stride
                horz_end=horz_start+k_w
                #now slicing the particular matrix so that we can cross_correlation
                A_slice=A_prev_pad[:,vert_start:vert_end,horz_start:horz_end,:]
                """
                the shape of the A_slice_slice is (m,k_h,k_w,k_d) and the shape of Kernel is (k_h,k_w,k_d,k_c)
                so as there is a mismatch in the shape in order to broad cast during matrix multiplication
                the problem lies in the number of dimensions for broadcasting as the A_prev_slice is lacking by 1 dimension
                we add a new dimension in A_prev_slice so its dim is (m,k_h,k_w,k_d,1)
                the last dim in A_prev_slice will broadcast to the k_c in kernel
                so the dimension in after matrix multiplication will be (m,k_h,k_w,k_d,k_c)
                then when we need to decide through which axis we will sum the matrix
                we will sum through row, column and the depth so the dimension of the matrix after one multiplied sum over r,c and d is (m,1,k_c)
                now as this was the dim when it was for one convo we will go through n_h and n_w
                So the dim of the matrix will result into (m,n_h,n_w,n_c) and this is what we want"""

                Z[:,i,j,:]=np.sum(A_slice[:,:,:,:,np.newaxis]*self.kernels,axis=(1,2,3))+self.bias
        return Z
