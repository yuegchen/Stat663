# reload_ext cython
# %%cython -a

import cython


import numpy as np
from libc.math cimport sqrt
from cpython cimport array
# import array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def plsr_cython(double[:,:]X,double[:,:]Y,int factors,double thre,double[:,:] T,double[:,:] U,double[:,:] P,double[:,:] Q,double[:,:] W,double[:] B):
    
    cdef int rows,colsX,colsY
    cdef double b,sum_,t_norm,norm
    cdef double[:] t,u,w,q,p,temp
    cdef double[:,:] E,F
    rows=X.shape[0]
    colsX=X.shape[1]
    colsY=Y.shape[1]
    E,F=X.copy(),Y.copy()
    
    for factor in range(factors):
        t=E.copy()[:,0]
        u=F.copy()[:,0]
        w=np.zeros(colsX)#array.array('d', [0,0,0])
        q=np.zeros(colsY)#array.array('d', [0,0])
        p=np.zeros(colsX)#array.array('d', [0,0,0])
        sum_=0
        for i in range(rows):
            sum_+=t[i]*t[i]
        t_norm=sqrt(sum_)
        
        while t_norm>1e-7:      
            temp=t.copy()
            for j in range(colsX):
                w[j]=0
                for i in range(rows):
                    w[j]+=E[i,j]*u[i]

            sum_=0
            for i in range(rows):
                sum_+=u[i]*u[i]

            for i in range(colsX):
                w[i]=w[i]/sum_

            sum_=0
            for i in range(colsX):
                sum_+=w[i]*w[i]
            norm=sqrt(sum_)
            
            for i in range(colsX):
                w[i]=w[i]/norm

            for i in range(rows):
                t[i]=0
                for j in range(colsX):
                    t[i]+=E[i,j]*w[j]

            sum_=0
            for i in range(colsX):
                sum_+=w[i]*w[i]  

            for i in range(rows):
                t[i]=t[i]/sum_
 
            for j in range(colsY):
                q[j]=0
                for i in range(rows):
                    q[j]+=F[i,j]*t[i]
            sum_=0
            for i in range(rows):
                sum_+=t[i]*t[i]  
            for i in range(colsY):
                q[i]=q[i]/sum_
            
            sum_=0
            for i in range(colsY):
                sum_+=q[i]*q[i]
            norm=sqrt(sum_)

            for i in range(colsY):
                q[i]=q[i]/norm
            

            for i in range(rows):
                u[i]=0
                for j in range(colsY):
                    u[i]+=F[i,j]*q[j]
            
            sum_=0
            for i in range(colsY):
                sum_+=q[i]*q[i]  

            for i in range(rows):
                u[i]=u[i]/sum_

            for i in range(rows):
                temp[i]-=t[i]
            
            sum_=0
            for i in range(rows):
                sum_+=temp[i]*temp[i]
            t_norm=sqrt(sum_)
            
        for j in range(colsX):
            p[j]=0
            for i in range(rows):
                p[j]+=E[i,j]*t[i]
                
        sum_=0
        for i in range(rows):
            sum_+=t[i]*t[i]  
        for i in range(colsX):
            p[i]=p[i]/sum_

        b=0
        for i in range(rows):
            b+=u[i]*t[i]/sum_
        sum_=0
        for i in range(colsX):
            sum_+=p[i]*p[i]
        norm=sqrt(sum_)
        
        for i in range(rows):
            t[i]=t[i]*norm
        for i in range(rows):
            w[i]=w[i]*norm
        for i in range(rows):
            p[i]=p[i]/norm

        for i in range(rows):
            for j in range(colsX):
                E[i,j]-=t[i]*p[j]
            for j in range(colsY):
                F[i,j]-=b*t[i]*q[j]

        T[:,factor]=t
        U[:,factor]=u
        P[:,factor]=p
        Q[:,factor]=q
        W[:,factor]=w
        B[factor]=b
        
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def predict_cython(double[:,:]W,double[:,:]Q,double[:,:]P,double[:,:]X,double[:,:]Y_,double[:]B,int factors):
    cdef double[:] t
    cdef int rows,colsX,colsY
    rows=X.shape[0]
    colsX=X.shape[1]
    colsY=Y_.shape[1]
    for i in range(factors):

        t=array.array('d', [0,0,0,0])
        for k in range(rows):
            for j in range(colsX):
                t[k]+=X[k,j]*W[j,i]
        for k in range(rows):
            for j in range(colsX):
                X[k,j]-=t[k]*P[j,i]
            for j in range(colsY):
                Y_[k,j]+=B[i]*t[k]*Q[j,i]

    return Y_
    