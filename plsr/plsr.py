import numpy as np
from numpy import linalg as LA

    
def plsr(X,Y,factors,thre,T,U,P,Q,W,B):
    rows=len(X)
    colsX=len(X[0])
    colsY=len(Y[0])
    E,F=X.copy(),Y.copy()
    
    for factor in range(factors):
        t=E[:,0]
        u=F[:,0]
        w=np.ones((colsX))
        q=np.ones((colsY))
        t_norm=LA.norm(t)
        
        while t_norm>1e-7:
            
            temp=t.copy()
            w=u.T@E/(u@u.T)
            w=w/LA.norm(w)
            t=E@w.T/(w@w.T)
            q=t@F/(t@t.T)
            q=q/LA.norm(q)
            u=F@q.T/(q@q.T)
            t_norm=LA.norm(t-temp)
        
        p=t@E/(t@t.T)
        t=t*LA.norm(p)
        w=w*LA.norm(p)
        p=p/LA.norm(p)
        
        b=u@t.T/(t@t.T)
        F_=F.copy()
        for i in range(len(t)):
            for j in range(len(p)):
                E[i,j]-=t[i]*p[j]
            for j in range(len(q)):
                F_[i,j]-=b*t[i]*q[j]
        T[:,factor]=t
        U[:,factor]=u
        P[:,factor]=p
        Q[:,factor]=q
        W[:,factor]=w
        B[factor]=b
        
        if LA.norm(F_)<thre or abs(LA.norm(F_)-LA.norm(F))<1e-10:
            return #already converged
        F=F_

def predict(W,Q,P,X,Y_,B,factors):
    
    for i in range(factors):
        t=X@W[:,i]
        for k in range(len(t)):
            for j in range(len(P)):
                X[k,j]-=t[k]*P[j,i]
            for j in range(len(Q)):
                Y_[k,j]+=B[i]*t[k]*Q[j,i]
    return Y_
    