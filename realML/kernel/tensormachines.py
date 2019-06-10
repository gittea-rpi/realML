import numpy as np
from .sfo import SFO
from time import time
from scipy.optimize import minimize

# NB: binary classification is done with +/- labels

def tm_preprocess(X, colnorms=None):
    """
    Preprocessing that seems to make TM more accurate:
        normalize each column so training data has length 1 (use same normalization constants for training and test)
        normalize each row to have length 1 (so normalization constant differs at test time)
    Inputs:
        X - feature matrix, rows are instances
        colnorms - vector containing the norm of each colum of the training matrix
    Outputs:
        if colnorms is None (training):
            Xnormalized - the normalized training data
            colnorms - the vector containing the norm of each column of the training matrix
        if colnorms is set (testing):
            Xnormalized - the normalized test data
    """

    returnargs = 1
    if colnorms is None:
        # Train
        colnorms = np.sqrt(np.sum(X*X, axis=0))
        returnargs = 2
    
    Xnormalized = np.copy(X)
    Xnormalized[:, colnorms > 0] = Xnormalized[:, colnorms > 0] / colnorms[colnorms > 0]
        
    rownorms = np.sqrt(np.sum(Xnormalized*Xnormalized, axis=1))
    Xnormalized = Xnormalized / rownorms[:, np.newaxis]

    if returnargs == 1:
        return Xnormalized
    elif returnargs == 2:
        return (Xnormalized, colnorms)
    

def tm_predict(w0, X, q, r, type):
    """
    Returns predicted values based on a learned tensor machine
    Inputs:
        w0 - TM factors
        X,q,r,type - see the description of tm_fit
    Outputs:
        z - predictions for each row in X
    """

    (n,d) = X.shape
    r_vec = np.concatenate(([1], (q-1)*[r]))
    b = w0[0]
    w = w0[1:]
    w = np.reshape(w, (d, len(w)//d))

    acc_sum = 0
    w_offset = 0

    Z = b*np.ones((n,1))

    for i in range(q):
        for j in range(r_vec[i]): 
            # the vectors whose outer product form the jth rank-one term in the
            # outer product of the coefficients for the degree i+1 term
            # d-by-i matrix
            W = w[:, w_offset:(w_offset + i + 1)]
            XW = X.dot(W) # n-by-(i+1)
            prodXW = np.prod(XW, axis=1) # n-by-1
            prodXW = prodXW[:, np.newaxis]
            
            Z = Z + prodXW # n-by-1
            w_offset = w_offset + i + 1 

    if type.upper() == 'REGRESSION':
        return Z
    elif type.upper() == 'BC':
        return np.sign(Z)
    
def tm_f_df(w0, X, y, q, r, type, gamma):
    """
    Computes the TM objective value and gradient for scipy's optimization functions
    Inputs:
        w0 - TM factors
        X,y,q,r,type,gamma - see the description of tm_fit
    Outputs:
        f - function value
        df - gradient of TM factors
    """

    (n,d) = X.shape
    r_vec = np.concatenate(([1], (q-1)*[r]))
    b = w0[0]
    w = w0[1:]
    w = np.reshape(w, (d, len(w)//d))
    nw = w.shape[1]

    acc_sum = 0
    w_offset = 0

    Z = b*np.ones((n,1))
    bl = np.zeros((n, nw))

    for i in range(q):
        for j in range(r_vec[i]): 
            # the vectors whose outer product form the jth rank-one term in the
            # outer product of the coefficients for the degree i+1 term
            # d-by-i matrix
            W = w[:, w_offset:(w_offset + i + 1)]
            XW = X.dot(W) # n-by-(i+1)
            prodXW = np.prod(XW, axis=1) # n-by-1
            prodXW = prodXW[:, np.newaxis] # make it a column vector
            bl[:, w_offset:(w_offset+i+1)] = prodXW / XW
            
            Wsquared = W*W
            norm_squares = np.sum(Wsquared, axis=0) # 1-by-(i+1)
            acc_sum = acc_sum + np.sum(norm_squares)

            Z = Z + prodXW # n-by-1
            w_offset = w_offset + i + 1 

    f = 0
    diff = np.empty_like(Z)
    if type.upper() == 'REGRESSION':
        diff = Z - y;
        f = np.sum(diff*diff)/n/2
    elif type.upper() == 'BC':
        eyz = np.exp(-y*Z);
        diff = -y*eyz/(1+eyz)
        f = np.mean(np.log(1 + eyz))
    f = f + gamma*acc_sum/2;

    df = np.empty_like(w0)
    df[0] = np.mean(diff)
    df_w = X.transpose().dot(diff*bl)
    df_w = df_w + gamma*w;
    df[1:] = np.reshape(df_w, (len(w0)-1,))

    return (f, df)

def tm_f_df_sub(w0, indices, X, y, q, r, type, gamma):
    """
    Computes the TM objective value and gradient for SFO solver
    Inputs:
        w0 - TM factors
        indices - list of indexes into the training data defining this minibatch
        X,y,q,r,type,gamma - see the description of tm_fit
    Outputs:
        f - function value
        df - gradient of TM factors
    """
    minibatchX = X[indices, :]
    minibatchy = y[indices, :]
    return tm_f_df0(w0, X, y, q, r, type, gamma)

def tm_f_df0(w0, X, y, q, r, type, gamma):
    """
    Computes the TM objective value and gradient for SFO
    Inputs:
        w0 - TM factors
        X,y,q,r,type,gamma - see the description of tm_fit
    Outputs:
        f - function value
        df - gradient of TM factors
    """

    (n,d) = X.shape
    gamma = n*gamma
    r_vec = np.concatenate(([1], (q-1)*[r]))
    b = w0[0]
    w = w0[1:]
    w = np.reshape(w, (d, len(w)//d))
    nw = w.shape[1]

    acc_sum = 0
    w_offset = 0

    Z = b*np.ones((n,1))
    bl = np.empty((n, nw))

    for i in range(q):
        for j in range(r_vec[i]): 
            # the vectors whose outer product form the jth rank-one term in the
            # outer product of the coefficients for the degree i+1 term
            # d-by-i matrix
            W = w[:, w_offset:(w_offset + i + 1)]
            XW = X.dot(W) # n-by-(i+1)
            prodXW = np.prod(XW, axis=1) # n-by-1
            prodXW = prodXW[:, np.newaxis]
            if i == 0: # dealing with the linear term
                bl[:, w_offset:(w_offset + i + 1)] = 1
            else:
                for l in range(i+1):
                    idx = np.setdiff1d([j for j in range(i+1)], l)
                    bl[:, w_offset+l] = np.prod(XW[:, idx]*XW[:, idx])
            
            Wsquared = W*W
            norm_squares = np.sum(Wsquared, axis=0) # 1-by-(i+1)
            acc_sum = acc_sum + np.sum(norm_squares)

            Z = Z + prodXW # n-by-1
            w_offset = w_offset + i + 1 

    f = 0
    diff = np.empty_like(Z)
    if type.upper() == 'REGRESSION':
        diff = Z - y;
        f = np.sum(diff*diff)/2
    elif type.upper() == 'BC':
        eyz = np.exp(-y*Z);
        diff = -y*eyz/(1+eyz)
        f = np.sum(np.log(1 + eyz))
    f = f + gamma*acc_sum/2;

    df = np.empty_like(w0)
    df[0] = np.sum(diff)
    df_w = X.transpose().dot(diff*bl)
    df_w = df_w + gamma*w;
    df[1:,0] = np.reshape(df_w, (len(w0)-1,))

    return (f, df)

def tm_fit(X, y, type, r, q, gamma, solver, epochs, alpha, verbosity='minimal', seed=0):
    """
    Inputs:
        X, y: feature matrix and target vector (numpy arrays)
        type: 'regression' or 'bc' for binary classification
        r: rank parameter
        q: degree of polynomial used
        gamma: regularization parameter
        solver: 'LBFGS' or 'SFO'
        epochs: maxiterations for L-BFGS or number of SFO epochs
        alpha: scaling factor of the initial weights
        verbosity: 'off', 'minimal', 'all'
        seed: seed for random number generation

    Outputs:
        w - factors used in the TM model
        z - predictions of X based on w
    """
    (n,d) = X.shape
    np.random.seed(seed)

    nv = 1 + d + ((q-1)*(q+2)*r*d)//2; # how many variables in total are in the factorization
    w0 = alpha*np.random.randn(nv,1) # set initial weights
    w = np.empty_like(w0)

    if solver.upper() == "LBFGS":

        options = {'maxiter' : epochs }
        res = minimize(tm_f_df, w0, args=(X,y,q,r,type,gamma), method="L-BFGS-B", jac=True, tol=1e-8, options=options)
        w = res.x

    elif solver.upper() == "SFO":

        N = max(30, int(np.floor(np.sqrt(n)/10))) # number of minibatches
        minibatch_indices = list()
        randp = np.array(np.random.permutation(n))
        for i in range(N):
            minibatch_indices.append(randp[i:n:N])

        optimizer = SFO(tm_f_df_sub, w0, minibatch_indices, args=(X,y,q,r,type,gamma))

        if verbosity.upper() == "OFF":
            optimizer.display = 0
        elif verbosity.upper() == "MINIMAL":
            optimizer.display = 1
        elif verbosity.upper() == "ALL":
            optimizer.display = 2

        w = optimizer.optimize(epochs)
        opt_outputs = optimizer;

    #else:
    #    print("Enter a valid solver! scipy's LBFGS and SFO are supported so far")

    z = tm_predict(w0, X, q, r, type)
    return (w, z)

def tm_solver(Xtrain, ytrain, Xtest, ytest, type, options):
    """
    Takes an input a training and test set and trains tensor machine then evaluates test error
    Inputs:
        Xtrain, ytrain - training features and targets
        Xtest, ytest - test features and targets
        type - 'regression' or 'bc' (binary classification)
        options - dictionary containing options for tensor machines (see tm_fit description for more information)
    Outputs:
        error_test, error_train: test and training errors (misclassification rate for bc, relative norm for regression)
    """

    (n,d) = Xtrain.shape
    ntest = Xtest.shape[0]

    #print("running tensor machine training")
    #print("data size: %d by %d" % (n,d))
    #print("parameters: degree(%d) rank(%d) solver(%s) gamma(%e) maxIter(%d) alpha(%f)" % 
    #      (options['q'], options['r'], options['solver'], options['gamma'], 
    #       options['maxIter'], options['alpha']))

    timeStart = time()
    (w, predtrain) = tm_fit(Xtrain, ytrain, type, options['r'], 
            options['q'], options['gamma'], options['solver'], 
            options['maxIter'], options['alpha'], options['verbosity'])
    timeEnd = time()
    #print("Finished training in %d seconds" % (timeEnd - timeStart))
    
    predtest = tm_predict(w, Xtest, options['q'], options['r'], type)

    error_train = 1
    error_test = 1
    if type.upper() == 'BC':
        predtrain = np.sign(predtrain)
        predtest = np.sign(predtest)
        error_train = 1 - np.mean(predtrain == ytrain)
        error_test = 1 - np.mean(predtest == ytest)
    elif type.upper() == 'REGRESSION':
        error_train = norm(ytrain - predtrain)/norm(ytrain)
        error_test = norm(ytest - predtest)/norm(ytest)

    #print('Training error: %f\n Testing error: %f' % (error_train, error_test))


