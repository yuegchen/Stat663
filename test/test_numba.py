from plsr import plsr_numba,predict_numba
import numpy as np

X = np.array([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]])
Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
rows=len(X)
colsX=len(X[0])
colsY=len(Y[0])
factors=2
T=np.zeros((rows,factors))
U=np.zeros((rows,factors))
P=np.zeros((colsX,factors))
Q=np.zeros((colsY,factors))
W=np.zeros((colsX,colsX))
B=np.zeros((colsX))
Y_=np.zeros((rows,colsY))
plsr_numba(X,Y,factors,1e-7,T,U,P,Q,W,B)
print(predict_numba(W,Q,P,X,Y_,B,factors))


