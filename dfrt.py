"""
Script of calculation Discrete Fractional Fourier Transform matrices for a given 
dimension N and fraction value a
"""

import numpy as np
from scipy import linalg
import math

def dfrtmtrx(N, a):
    # Approximation order
    app_ord = 2
    Evec = _dis_s(N,app_ord)
    
    even = 1 - (N%2)
    
    l = np.array(list(range(0,N-1)) + [N-1+even])
    
    f = np.diag(np.exp(-1j*math.pi/2*a*l))
    
    F= N**(1/2)*np.einsum("ij,jk,ni->nk", f, Evec.T, Evec, optimize=True)
    
    return F

def _dis_s(N,app_ord):
    
    S = _creates(N,app_ord)
    
    p = N
    r = math.floor(N/2)
    P = np.zeros((p,p))
    
    P[0,0] = 1
    even = 1 - (p%2)
    
    for i in range(1,r-even+1):
        P[i,i] = 1/(2**(1/2))
        P[i,p-i] = 1/(2**(1/2))
        
    if even:
        P[r,r] = 1
        
    for i in range(r+1,p):
        P[i,i] = -1/(2**(1/2))
        P[i,p-i] = 1/(2**(1/2))
    
    
    CS = np.einsum("ij,jk,ni->nk", S, P.T, P, optimize=True)
    
    C2 = CS[0:math.floor(N/2+1), 0:math.floor(N/2+1)]
    S2 = CS[math.floor(N/2+1):N, math.floor(N/2+1):N]
    
    ec, vc = linalg.eig(C2)
    # idx = np.argsort(ec)
    # ec = ec[idx]
    # vc = vc[:,idx]
    
    es, vs = linalg.eig(S2)
    # idx = np.argsort(es)
    # es = es[idx]
    # vs = vs[:,idx]
    
    qvc = np.vstack((vc, np.zeros([math.ceil(N/2-1), math.floor(N/2+1)])))
    SC2 = P@qvc # Even Eigenvector of S
    
    qvs = np.vstack((np.zeros([math.floor(N/2+1), math.ceil(N/2-1)]),vs))
    SS2 = P@qvs # Odd Eigenvector of S
    
    idx = np.argsort(-ec)
    SC2 = SC2[:,idx]
    
    idx = np.argsort(-es)
    SS2 = SS2[:,idx]
    
    if N%2 == 0:
        S2C2 = np.zeros([N,N+1])
        SS2 = np.hstack([SS2, np.zeros((SS2.shape[0],1))])
        S2C2[:,range(0,N+1,2)] = SC2;
        S2C2[:,range(1,N,2)] = SS2
        S2C2 = np.delete(S2C2, (N-1), axis=1)
    else:
        S2C2 = np.zeros([N,N])
        S2C2[:,range(0,N+1,2)] = SC2;
        S2C2[:,range(1,N,2)] = SS2
    
    Evec = S2C2 
    
    return Evec
    
def _creates(N,app_ord):
    # Creates S matrix of approximation order ord
    # When ord=1, elementary S matrix is returned
    
    app_ord = int(app_ord / 2) 

    s = np.concatenate((np.array([0, 1]), np.zeros(N-1-2*app_ord), np.array([1])))
    S = _cconvm(N,s) + np.diag((np.fft.fft(s)).real);
    
    return S
    
def _cconvm(N,s):
    # Generates circular Convm matrix
    M = np.zeros((N,N))
    dum = s
    for i in range(N):
        M[:,i] = dum
        dum = np.roll(dum,1)
        
    return M