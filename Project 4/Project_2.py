"""MATH96012 Project 2"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import time
from scipy.optimize import minimize
from a1 import lrmodel as lr #assumes that p2_templatedev.f90 has been compiled with: f2py -c p2_templatedev.f90 -m m1
# May also use scipy, scikit-learn, and time modules as needed f2py3 -c p2.f90 -m m1

def read_data(tsize=15000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 20000 matrix, X
    and the corresponding labels are stored in a 20000 element array, y.
    The final 20000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int) #rescale the image, convert the labels to integers between 0 and M-1)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def clr_test(X,y,X_test,y_test,bnd=1.0,l=0.0,input=(None)):
    """Train CLR model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    bnd: Constraint parameter for optimization problem
    lambda: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """

    n = X.shape[0]
    d = X.shape[1]
    d1 = X_test.shape[1]
    y = y%2
    fvec = np.random.randn(n+1)*0.1 #initial fitting parameters

    #set X,Y,....ect in fortan code
    lr.lr_x=X
    lr.lr_y=y
    lr.lr_l=l
    lr.te_x=X_test
    lr.te_y=y_test


    #Add code to train CLR model and evaluate testing test_error

    cons=[(-bnd,bnd)]*n+[(None,None)]
    parameters=(d,n)
    result = minimize(lr.clrmodel,fvec, jac=True,args=parameters, method='L-BFGS-B',bounds=cons)
    error = lr.clr_test_error(result.x,d1,n)

    fvec_f = result.x #Modify to store final fitting parameters after training
    test_error = error #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (None) #output tuple, modify as needed




    return fvec_f,test_error,output
#--------------------------------------------

def mlr_test(X,y,X_test,y_test,m=2,bnd=1.0,l=0.0,input=(None)):
    """Train MLR model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of classes
    bnd: Constraint parameter for optimization problem
    lambda: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """
    #calculate variables
    d = X.shape[1]
    d1 = X_test.shape[1]
    n = X.shape[0]
    y = y%m
    y_test = y_test%m

    fvec = np.random.randn((m-1)*(n+1))*0.1 #initial fitting parameters

    lr.lr_x=X
    lr.lr_y=y
    lr.lr_l=l
    lr.te_x=X_test
    lr.te_y=y_test

#code to train MLR model and evaluate testing error, test_error
    cons=[(-bnd,bnd)]*n*(m-1)+[(None,None)]*(m-1)
    parameters=(n,d,m)
    result = minimize(lr.mlrmodel,fvec, jac=True,args=parameters, method='L-BFGS-B',bounds=cons)
    error = lr.mlr_test_error(result.x,n,d1,m)



    fvec_f = result.x #Modify to store final fitting parameters after training
    test_error = error #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (None) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def lr_compare():
    """ Analyze performance of MLR and neural network models
    on image classification problem
    Add input variables and modify return statement as needed.
    Should be called from name==main section below
    """
    X,y,X_test,y_test=read_data(tsize=15000)
    m=3
    timeMPL=np.zeros(10)
    timeMLR=np.zeros(10)
    darray=np.zeros(10)
    for d in range(1,10):
        X_train=X[:,:d*100]
        Y_train=y[:d*100]%m
        startmpl = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='lbfgs', verbose=10, tol=1e-4, random_state=1,learning_rate_init=.1)
        mlp.fit(X_train.T, Y_train)
        endmpl = time.time()
        timeMPL[d]=endmpl - startmpl
        darray[d]=d*100+1



        startmlr = time.time()
        stuff=mlr_test(X_train,Y_train,X_test,y_test,m=m,bnd=1.0,l=0.0,input=(None))
        endmlr = time.time()
        timeMLR[d]=endmlr - startmlr

    plt.figure()
    plt.plot(darray,timeMPL, label='MLP')
    plt.plot(darray,timeMLR, label='MLR')
    plt.legend()
    plt.xlabel("Training size d")
    plt.ylabel("run time (seconds)")
    plt.title("Matthew Cowley, lr_compare, Run time against training size")

    return None
#--------------------------------------------

def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------







if __name__ == '__main__':
    output = lr_compare()
