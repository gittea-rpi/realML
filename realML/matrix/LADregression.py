from scipy.linalg import eig, qr, solve, norm
from scipy.sparse import identity as speye, dia_matrix, coo_matrix
from numpy.random import rand, choice, randn
import numpy as np

def LAD(X, y, eps, maxIters):
    """Solves LAD on the system Xb = y using eps relative error
    as stopping criterion, and at most maxIter iterations
    """
    n,m = X.shape
    weights = np.ones((n,1))
    delta = .001;
    
    for curiter in np.arange(maxIters):
       b = solve(X.T.dot(weights*X), X.T.dot(weights*y))
       resids = np.abs(y - X.dot(b))
       weights = 1/np.maximum(delta, resids)
       if (curiter > 1 and norm(b - oldb)/norm(b) < eps):
           break
       if curiter > 1:
           #print(norm(b-oldb)/norm(b))
           pass
       oldb = b
    return b
    
def CountSketch(n, c):
    """Returns a sparse CountSketch matrix from n dimensions to c"""
    h = choice(c, n)
    s = choice([1.0, -1.0], n)
    Pi = coo_matrix((s, (h, np.arange(n))), shape=(c, n))
    return Pi

def SparseCauchy(n, c):
    """Returns a sparse Cauchy transform from n dimensions to c"""
    h = choice(c,n)
    s = np.tan(np.pi*(rand(n) - 0.5))
    Pi = coo_matrix((s, (h, np.arange(n))), shape=(c,n))
    return Pi

def generateWellConditionedBasis(A, r):
    """Computes an l1-well conditioned basis for the span of A by
    using a sketch of the rowspace of A to get an approximate 
    column orthogonalizer for A. Assumes A is tall-and-skinny.
    The sketch is taken from Magdon-Ismail, Gittens 2018
    """

    n, d = A.shape
    r2 = r - d

    sf = np.sqrt(d)*np.log(d)
    cauchyVariates = np.tan(np.pi*(rand(n,1) - 0.5))
    X = CountSketch(n, r).dot(A)
    eigenvals, V = eig(X.T.dot(X))
    Pi1 = sf*(V*np.sqrt(eigenvals)).dot(V.T)
    Pi2 = SparseCauchy(n, r2).dot(A)

    Pi = np.concatenate((Pi1, Pi2))
    _, Rfactor = qr(Pi, mode='economic')
    return solve(Rfactor.T, A.T).T

def sampleProblem(X, y, probs):
    """Returns a sample of rescaled rows of X and y, where each row is sampled
    as a Bernoulli with mean given by the corresponding entry of probs,
    and rescaled accordingly
    """
    n, d = X.shape
    rowindices, *_ = np.where( rand(n) <= probs)
    weights = np.expand_dims(1/probs[rowindices], axis=1)

    Xs = weights*X[rowindices,:]
    ys = weights*y[rowindices,:]
    return (Xs, ys)

def coresetLAD(X, y, U, r, eps, maxIters):
    """Computes a coreset of [X,y] of around size r using 
    the basis U, then uses IRLS to solve LAD to eps relative
    accuracy using at most maxIters iterations
    """
    probs = np.minimum(1.0, r*np.sum(np.abs(U), 1)/np.sum(np.abs(U)))
    Xs, ys = sampleProblem(X, y, probs)
    b = LAD(Xs, ys, eps, maxIters)
    return b

if __name__ == "__main__":
    # Test the functions
    from timeit import default_timer as timer

    d = 50
    n = d**3
    r = 3*d;
    
    sf = 1/np.sqrt(n)
    Xtrain = np.zeros((n,d))
    ytrain = np.zeros((n,1))
    alpha = 20;
    
    for index in np.arange(d):
        startrow = d*index
        endrow = d*(index+1)
        indicatorvec = sf*rand(d,1)
        indicatorvec[index] = 1
        Xtrain[startrow:endrow, :] = indicatorvec.dot(indicatorvec.T) + \
            (np.eye(d) - indicatorvec.dot(indicatorvec.T)).dot(randn(d,d)).dot(np.eye(d) - np.ones((d,1)).dot(np.ones((1,d)))/d)
        ytrain[startrow:endrow,:] = alpha*indicatorvec
    
    Xtrain[d**2:,:] = sf*randn(n-d**2, d).dot(np.eye(d) - np.ones((d,1)).dot(np.ones((1,d))/d))
    ytrain[d**2:,:] = sf*randn(n-d**2, 1)

    maxIters = 300
    eps = 1e-6
    stoppingTol = eps*norm(ytrain, 1)/(np.sqrt(n)*norm(Xtrain))
    #print('finished setting up training dataset')
    
    start = timer()
    irlssoln = LAD(Xtrain, ytrain, stoppingTol, maxIters)
    end = timer()
    irlserr = norm(Xtrain.dot(irlssoln) - ytrain, 1)
    #print("Obj error of full problem is %f and elapsed time is %f (s)" % (irlserr, end - start))

    start = timer()
    probs = r/n * np.ones(n)
    (Xs, ys) = sampleProblem(Xtrain, ytrain, probs)
    unifsoln = LAD(Xs, ys, stoppingTol, maxIters)
    end = timer()
    reluniferr = norm(Xtrain.dot(unifsoln) - ytrain, 1)/irlserr
    #print("Rel obj error of unif sampled problem is %f and elapsed time is %f (s)" % (reluniferr, end-start))

    start = timer()
    U = generateWellConditionedBasis(np.concatenate((Xtrain, ytrain), axis=1), r)
    MGsoln = coresetLAD(Xtrain, ytrain, U, r, stoppingTol, maxIters)
    end = timer()
    relMGerr = norm(Xtrain.dot(MGsoln) - ytrain, 1)/irlserr
    #print("Rel obj error of MG sampled problem is %f and elapsed time is %f (s)" % (relMGerr, end-start))
