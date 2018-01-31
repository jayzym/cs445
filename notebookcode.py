
# coding: utf-8

# $\newcommand{\xv}{\mathbf{x}}
# \newcommand{\Xv}{\mathbf{X}}
# \newcommand{\yv}{\mathbf{y}}
# \newcommand{\zv}{\mathbf{z}}
# \newcommand{\av}{\mathbf{a}}
# \newcommand{\Wv}{\mathbf{W}}
# \newcommand{\wv}{\mathbf{w}}
# \newcommand{\tv}{\mathbf{t}}
# \newcommand{\Tv}{\mathbf{T}}
# \newcommand{\muv}{\boldsymbol{\mu}}
# \newcommand{\sigmav}{\boldsymbol{\sigma}}
# \newcommand{\phiv}{\boldsymbol{\phi}}
# \newcommand{\Phiv}{\boldsymbol{\Phi}}
# \newcommand{\Sigmav}{\boldsymbol{\Sigma}}
# \newcommand{\Lambdav}{\boldsymbol{\Lambda}}
# \newcommand{\half}{\frac{1}{2}}
# \newcommand{\argmax}[1]{\underset{#1}{\operatorname{argmax}}}
# \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}}$

# # Assignment 1: Linear Regression

# *Type your name here and DELETE ALL TEXT PROVIDED HERE THAT ARE INSTRUCTIONS TO YOU*

# ## Overview

# Describe the objective of this assignment, and very briefly how you accomplish it.  Say things like "linear model", "samples of inputs and known desired outputs" and "minimize the sum of squared errors". DELETE THIS TEXT AND INSERT YOUR OWN.

# ## Method

# Define in code cells the following functions as discussed in class.  Your functions' arguments and return types must be as shown here.
# 
#   * ```model = train(X, T)```
#   * ```predict = use(model, X)```
#   * ```error = rmse(predict, T)```
#   
# Let ```X``` be a two-dimensional matrix (```np.array```) with each row containing one data sample, and ```T``` be a two-dimensional matrix of one column containing the target values for each sample in ```X```.  So, ```X.shape[0]``` is equal to ```T.shape[0]```.   
# 
# Function ```train``` must standardize the input data in ```X``` and return a dictionary with  keys named ```means```, ```stds```, and ```w```.  
# 
# Function ```use``` must also standardize its input data X by using the means and standard deviations in the dictionary returned by ```train```.
# 
# Function ```rmse``` returns the square root of the mean of the squared error between ```predict``` and ```T```.
# 
# Also implement the function
# 
#    * ```model = trainSGD(X, T, learningRate, numberOfIterations)```
# 
# which performs the incremental training process described in class as stochastic gradient descent (SGC).  The result of this function is a dictionary with the same keys as the dictionary returned by the above ```train``` function.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[26]:

def train(X, T):
    #Code taken from Professor Anderson's notebook
    means = X.mean(0)
    stds = X.std(0)
    Xs = (X - means) / stds
    Xs1 = np.insert(Xs, 0, 1, 1)               
    w = np.linalg.lstsq( Xs1.T @ Xs1, Xs1.T @ T)[0]
    
    #build return dictionary
    ret = {"means" : means,
           "stds" : stds,
           "w" : w
          }
    return ret


# In[75]:

def use(model, X):
    #Code taken from Professor Anderson's notebook
    newX = (X - model["means"]) / model["stds"]
    
    #insert columns into array
    newX = np.insert(newX, 0, 1, 1)
    
    #make prediction
    prediction = newX @ model["w"]
    
    return prediction


# In[154]:

def rmse(predict, T):
    #find average of the difference, square it and then square root
    return np.sqrt(np.mean((predict -  T)**2))
    


# In[160]:

def trainSGD(X, T, learningRate, numberOfIterations):
    
    w = np.zeros((2,1))

    # Collect the weights after each update in a list for later plotting. 
    # This is not part of the training algorithm
    ws = [w.copy()]

    xs = np.linspace(0, 10, 100).reshape((-1,1))
    xs1 = np.insert(xs, 0, 1, axis=1)
    
    newX = np.insert(X, 0, 1, axis=1)
    step = 0
    for iter in range(numberOfIterations):
        for n in range(learningRate):
        
            step += 1
        
            predicted = newX[n:n+1,:] @ w  # n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
            # Update w using negative derivative of error for nth sample
            w += eta * newX[n:n+1, :].T * (T[n:n+1, :] - predicted)
            ws.append(w.copy())
                        
    return w


# ## Examples

# In[155]:

import numpy as np

X = np.arange(10).reshape((5,2))
T = X[:,0:1] + 2 * X[:,1:2] + np.random.uniform(-1, 1,(5, 1))
print('Inputs')
print(X)
print('Targets')
print(T)


# In[156]:

model = train(X, T)
model


# In[157]:

predicted = use(model, X)
predicted


# In[158]:

rmse(predicted, T)


# In[147]:

modelSGD = trainSGD(X, T, 0.01, 100)
modelSGD


# In[8]:

predicted = use(modelSGD, X)
predicted


# In[9]:

rmse(predicted, T)


# ## Data

# Download ```energydata_complete.csv``` from the [Appliances energy prediction Data Set ](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) at the UCI Machine Learning Repository. Ignore the first column (date and time), use the next two columns as target variables, and use all but the last two columns (named rv1 and rv2) as input variables. 
# 
# In this section include a summary of this data, including the number of samples, the number and kinds of input variables, and the number and kinds of target variables.  Also mention who recorded the data and how.  Some of this information can be found in the paper that is linked to at the UCI site for this data set.  Also show some plots of target variables versus some of the input variables to investigate whether or not linear relationships might exist.  Discuss your observations of these plots.

# ## Results

# Apply your functions to the data.  Compare the error you get as a result of both training functions.  Experiment with different learning rates for ```trainSGD``` and discuss the errors.
# 
# Make some plots of the predicted energy uses and the actual energy uses versus the sample index.  Also plot predicted energy use versus actual energy use.  Show the above plots for the appliances energy use and repeat them for the lights energy use. Discuss your observations of each graph.
# 
# Show the values of the resulting weights and discuss which ones might be least relevant for fitting your linear model.  Remove them, fit the linear model again, plot the results, and discuss what you see.

# ## Grading
# 
# Your notebook will be run and graded automatically.  Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A1grader.tar) and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  You should see a perfect execution score of 70/70 if your functions are defined correctly. The remaining 30 points will be based on the results you obtain from the energy data and on your discussions.
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook.  It will include additional tests.  You need not include code to test that the values passed in to your functions are the correct form.  

# In[159]:

get_ipython().magic('run -i "A1grader.py"')


# ## Check-in

# Do not include this section in your notebook.
# 
# Name your notebook ```Lastname-A1.ipynb```.  So, for me it would be ```Anderson-A1.ipynb```.  Submit the file using the ```Assignment 1``` link on [Canvas](https://colostate.instructure.com/courses/41327).
# 
# Grading will be based on 
# 
#   * correct behavior of the required functions listed above,
#   * easy to understand plots in your notebook,
#   * readability of the notebook,
#   * effort in making interesting observations, and in formatting your notebook.

# ## Extra Credit

# Download a second data set and repeat all of the steps of this assignment on that data set.

# In[ ]:



