# # CSE474/574 - Programming Assignment 1


# In[ ]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pickle


# ## Part 1 - Linear Regression

# ### Problem 1 - Linear Regression with Direct Minimization

# In[ ]:


print('PROBLEM 1')
print('----------')


# In[ ]:


def learnOLERegression(X,y):
    # Inputs:   
    # d = Number of features = 64
    # N = Number of data of one feature = 442
    # X = N x d  
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    x_tran = X.T # d x N
    xdotx = np.dot(x_tran,X) # d x d
    inv = np.linalg.inv(xdotx) # dxd
    q = np.dot(x_tran,y)
    w = np.dot(inv,q) # dx1
    return w


# In[ ]:


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # rmse = scalar value
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    #rmse = 0
    for i in range(ytest.shape[1]):
        w_x = np.dot(w.T,Xtest.T) # 1 x N
        d = ytest - w_x.T # N x 1
        b = np.square(d) # N x 1
    
    c = np.sum(b) #scalar
    e = c/Xtest.shape[0]
    rmse = np.sqrt(e)
    return rmse


# In[ ]:


Xtrain,ytrain,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding='latin1')  
a = np.array([[1., 2.], [3., 4.]])

# add intercept
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[ ]:


x1 = np.ones((len(Xtrain),1))
x2 = np.ones((len(Xtest),1))


# In[ ]:


Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
print(Xtrain_i.shape)
print(Xtest_i.shape)


# In[ ]:


w = learnOLERegression(Xtrain,ytrain)
w_i = learnOLERegression(Xtrain_i,ytrain)
print(w.shape)
print(w_i.shape)
print(Xtest.shape[1])


# In[ ]:


w = learnOLERegression(Xtrain,ytrain)
w_i = learnOLERegression(Xtrain_i,ytrain)
print(w.shape)
print(w_i.shape)

rmse = testOLERegression(w,Xtrain,ytrain)
rmse_i = testOLERegression(w_i,Xtrain_i,ytrain)
print('RMSE without intercept on train data - %.2f'%rmse)
print('RMSE with intercept on train data - %.2f'%rmse_i)

rmse = testOLERegression(w,Xtest,ytest)
rmse_i = testOLERegression(w_i,Xtest_i,ytest)
print('RMSE without intercept on test data - %.2f'%rmse)
print('RMSE with intercept on test data - %.2f'%rmse_i)


# ### Problem 2 - Linear Regression with Gradient Descent

# In[ ]:


print('PROBLEM 2')
print('----------')


# In[ ]:


def regressionObjVal(w, X, y):

    # compute squared error (scalar) with respect
    # to w (vector) for the given data X and y      
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar value

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    
    x_w = np.dot(X,w) # Nx1
    term_2 = np.subtract(y,x_w)
    term_1 = np.subtract(y,x_w.T)
    dot_prod = np.dot(term_1,term_2)[0][0]
    error = dot_prod/2
    return error


# In[ ]:


def regressionGradient(w, X, y):

    # compute gradient of squared error (scalar) with respect
    # to w (vector) for the given data X and y   
    
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # gradient = d length vector (not a d x 1 matrix)

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE 
    w = w.reshape(-1,1)
    x_tran = X.T
    term_2 = np.dot(x_tran,y) # Nx1
    xdotx = np.dot(x_tran,X)
    term_1 = np.dot(xdotx,w) 
    error_grad = np.subtract(term_1,term_2) 
    #error_grad = [i[0] for i in error_grad]
    #error_grad = np.array(error_grad)
    error_grad = error_grad.reshape(-1) # converting into vector
    return error_grad


# In[ ]:


Xtrain,ytrain,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding='latin1')   
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
args = (Xtrain_i,ytrain)
print(Xtrain.shape)
print(len(args))


# In[ ]:


opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))


# In[ ]:


soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args,method='CG', options=opts)


# In[ ]:


w = np.transpose(np.array(soln.x))
w = w[:,np.newaxis]
rmse = testOLERegression(w,Xtrain_i,ytrain)
print('Gradient Descent Linear Regression RMSE on train data - %.2f'%rmse)
rmse = testOLERegression(w,Xtest_i,ytest)
print('Gradient Descent Linear Regression RMSE on test data - %.2f'%rmse)


# ## Part 2 - Linear Classification

# ### Problem 3 - Perceptron using Gradient Descent

# In[ ]:


print('PROBLEM 3')
print('----------')


# In[ ]:


def predictLinearModel(w,Xtest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # Output:
    # ypred = N x 1 vector of predictions

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    N = len(Xtest)
    w_x = np.dot(w.T,Xtest.T)
    
    ypred = np.zeros([Xtest.shape[0],1])
    for i in range(0,N):
        if w_x[i] >= 0:
            ypred[i] = 1
        else: 
            ypred[i] = -1

    return ypred


# In[ ]:


def evaluateLinearModel(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # acc = scalar values

    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    ypred = predictLinearModel(w,Xtest)
    N = Xtest[0]
    for i in raneg(N):
        if ypred[i] == ytest[i]:
            acc += 1
    return acc


# In[ ]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)


# In[ ]:


args = (Xtrain_i,ytrain)
len(args)


# In[ ]:


opts = {'maxiter' : 50}    # Preferred value. 


# In[ ]:


w_init = np.zeros((Xtrain_i.shape[1],1))
soln = minimize(regressionObjVal, w_init, jac=regressionGradient, args=args,method='CG', options=opts)


# In[ ]:


w = np.transpose(np.array(soln.x))
w = w[:,np.newaxis]
print(w.shape)
print(len(Xtest))
print(Xtrain.shape)
print(Xtrain_i.shape)
print(ytrain.shape)


# In[ ]:


acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print(acc.shape)
print('Perceptron Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('Perceptron Accuracy on test data - %.2f'%acc)


# ### Problem 4 - Logistic Regression Using Newton's Method

# In[ ]:


print('PROBLEM 4')
print('----------')


# In[ ]:


def logisticObjVal(w, X, y):

    # compute log-loss error (scalar) with respect
    # to w (vector) for the given data X and y                               
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = scalar
    
    
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    error = 0
    return error


# In[ ]:


def logisticGradient(w, X, y):

    # compute the gradient of the log-loss error (vector) with respect
    # to w (vector) for the given data X and y  
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # error = d length gradient vector (not a d x 1 matrix)

    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    gradient = np.zeros((w.shape[0],))
    return gradient


# In[ ]:


def logisticHessian(w, X, y):

    # compute the Hessian of the log-loss error (matrix) with respect
    # to w (vector) for the given data X and y                               
    #
    # Inputs:
    # w = d x 1
    # X = N x d
    # y = N x 1
    # Output:
    # Hessian = d x d matrix
    
    if len(w.shape) == 1:
        w = w[:,np.newaxis]
    # IMPLEMENT THIS METHOD - REMOVE THE NEXT LINE
    hessian = np.eye(X.shape[1])
    return hessian


# In[ ]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

args = (Xtrain_i,ytrain)
opts = {'maxiter' : 50}    # Preferred value.    
w_init = np.zeros((Xtrain_i.shape[1],1))
soln = minimize(logisticObjVal, w_init, jac=logisticGradient, hess=logisticHessian, args=args,method='Newton-CG', options=opts)
w = np.transpose(np.array(soln.x))
w = np.reshape(w,[len(w),1])
acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print('Logistic Regression Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('Logistic Regression Accuracy on test data - %.2f'%acc)


# ### Problem 5 - Support Vector Machines Using Gradient Descent

# In[ ]:


print('PROBLEM 5')
print('----------')


# In[ ]:


def trainSGDSVM(X,y,T,eta=0.01):
    # learn a linear SVM by implementing the SGD algorithm
    #
    # Inputs:
    # X = N x d
    # y = N x 1
    # T = number of iterations
    # eta = learning rate
    # Output:
    # weight vector, w = d x 1
    
    # IMPLEMENT THIS METHOD
    w = np.zeros([X.shape[1],1])
    return w


# In[ ]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

args = (Xtrain_i,ytrain)
w = trainSGDSVM(Xtrain_i,ytrain,100,0.01)
acc = evaluateLinearModel(w,Xtrain_i,ytrain)
print('SVM Accuracy on train data - %.2f'%acc)
acc = evaluateLinearModel(w,Xtest_i,ytest)
print('SVM Accuracy on test data - %.2f'%acc)


# ### Problem 6 - Plotting decision boundaries

# In[ ]:


print('Problem 6')
print('---------')


# In[ ]:


def plotBoundaries(w,X,y):
    # plotting boundaries

    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    x1 = np.linspace(mn[1],mx[1],100)
    x2 = np.linspace(mn[2],mx[2],100)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.ravel()
    xx[:,1] = xx2.ravel()
    xx_i = np.concatenate((np.ones((xx.shape[0],1)), xx), axis=1)
    ypred = predictLinearModel(w,xx_i)
    ax.contourf(x1,x2,ypred.reshape((x1.shape[0],x2.shape[0])),alpha=0.3,cmap='cool')
    ax.scatter(X[:,1],X[:,2],c=y.flatten())


# In[ ]:


Xtrain,ytrain, Xtest, ytest = pickle.load(open('sample.pickle','rb')) 
# add intercept
Xtrain_i = np.concatenate((np.ones((Xtrain.shape[0],1)), Xtrain), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

# Replace next three lines with code for learning w using the three methods
w_perceptron = np.zeros((Xtrain_i.shape[1],1))
w_logistic = np.zeros((Xtrain_i.shape[1],1))
w_svm = np.zeros((Xtrain_i.shape[1],1))
fig = plt.figure(figsize=(20,6))

ax = plt.subplot(1,3,1)
plotBoundaries(w_perceptron,Xtrain_i,ytrain)
ax.set_title('Perceptron')

ax = plt.subplot(1,3,2)
plotBoundaries(w_logistic,Xtrain_i,ytrain)
ax.set_title('Logistic Regression')

ax = plt.subplot(1,3,3)
plotBoundaries(w_svm,Xtrain_i,ytrain)
ax.set_title('SVM')

