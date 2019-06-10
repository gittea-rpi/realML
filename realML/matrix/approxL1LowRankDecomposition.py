from scipy.linalg import solve, qr, svd, pinv, norm, orth
from numpy.random import rand
import numpy as np

def DenseCauchy(n, c):
    """Returns a dense Cauchy transform from n dimensions to c
    """
    return np.tan(np.pi*(rand(c,n)-0.5))

def FastL1LowRank(X, k, alpha = 5, repeats = 20):
    """Returns rank-k factorization of X = U*V' that is approximately
    optimal in the entrywise-l1 norm
    """

    (bestA,bestB) = L1LowRank(X, k, alpha)
    bestapproxerr = norm((bestA.dot(bestB) - X).flatten(), 1)

    for rep in range(repeats):
        (curA, curB) = L1LowRank(X, k, alpha)
        curapproxerr = norm((curA.dot(curB) - X).flatten(), 1)
        if curapproxerr < bestapproxerr:
            bestA = curA
            bestB = curB

    return (bestA, bestB)

def L1LowRank(X, k, alpha = 5):
    """Returns rank-k factorization of X = U*V' that is approximately
    optimal in the entrywise-l1 norm
    """
    n, m = X.shape
    s = alpha*k # k*log(k)
    T1 = DenseCauchy(n, s)
    T2 = DenseCauchy(n, s)
    S1 = DenseCauchy(m, s).T
    S2 = DenseCauchy(m, s).T

    L = T1.dot(X).dot(S1)
    R = T2.dot(X).dot(S2)
    Y = T1.dot(X).dot(S2)

    QL = orth(L)
    QRT= orth(R.T)

    Z = QL.T.dot(Y).dot(QRT)
    U,Sigma,VT = svd(Z)
    Uprime = QL.dot(U)
    VTprime = VT.dot(QRT.T)
    
    finalrank = min(k, len(Sigma) )
    Uprimek = Uprime[:, :finalrank]
    VTprimek = VTprime[:finalrank, :]

    A = X.dot(S1).dot(pinv(L)).dot(Uprimek)*np.sqrt(Sigma[:finalrank])
    B = np.sqrt(np.expand_dims(Sigma[:finalrank], axis=1))*VTprimek.dot(pinv(R)).dot(T2).dot(X)

    return (A,B)

if __name__ == "__main__":
    # construct a matrix whose best l1 approx has l1 error n, while best l2 approx has l1 error (n-1)^2
    # so should see this l1 method greatly outperforming l2 heuristic

    n = 200
    X = np.zeros((n,n))
    X[0,0] = n
    X[1:n, 1:n] = 1

    #(A,B) = FastL1LowRank(X, 1)
    (A,B) = L1LowRank(X, 1)
    CauchyXapprox = A.dot(B)

    cauchyerr = norm((CauchyXapprox - X).flatten(), 1)
    #print("Relative objective value using Cauchy transforms is %f" % (cauchyerr/n))

    U,S,VT = svd(X)
    l2approx = (U[:, :1]*S[:1]).dot(VT[:1,:])
    l2err = norm((l2approx - X).flatten(), 1)
    #print("Relative objective value using SVD is %f" % (l2err/n))

