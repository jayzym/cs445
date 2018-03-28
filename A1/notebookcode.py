
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

# Jared Zymbaluk

# ## Overview

# Describe the objective of this assignment, and very briefly how you accomplish it.  Say things like "linear model", "samples of inputs and known desired outputs" and "minimize the sum of squared errors". DELETE THIS TEXT AND INSERT YOUR OWN.

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

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:

def train(X, T):
    #Code taken from Professor Anderson's notebook
    means = X.mean(0)
    stds = X.std(0)
    Xs = (X - means) / stds
    Xs1 = np.insert(Xs, 0, 1, 1)               
    w = np.linalg.lstsq( Xs1.T @ Xs1, Xs1.T @ T)[0]
    
    #build return dictionary
    return {"means" : means,
             "stds" : stds,
             "w" : w
           }
    


# In[ ]:

def use(model, X):
    #Code taken from Professor Anderson's notebook
    newX = (X - model["means"]) / model["stds"]
    
    #insert columns into array
    newX = np.insert(newX, 0, 1, 1)
    
    #make prediction
    prediction = newX @ model["w"]
    
    return prediction


# In[ ]:

def rmse(predict, T):
    #find average of the difference, square it and then square root
    return np.sqrt(np.mean((predict -  T)**2))
    


# In[ ]:

def trainSGD(X, T, learningRate, numberOfIterations):
    #Code taken from Professor Anderson's notebook
    means = X.mean(0)
    stds = X.std(0)
    Xs = (X - means) / stds
    Xs1 = np.insert(Xs, 0, 1, axis=1)
    
    w = np.zeros((Xs1.shape[1],T.shape[1]))
    nOutputs = T.shape[1]
    nInputs = Xs1.shape[1]
    
    for iter in range(numberOfIterations):
        for n in range(len(X)):
            predicted = Xs1[n:n+1,:] @ w
            w += learningRate * Xs1[n:n+1, :].T * (T[n:n+1, :] - predicted)
        
                        
    return {"means" : means,
           "stds" : stds,
           "w" : w
          }


# ## Examples

# In[ ]:

import numpy as np

X = np.arange(10).reshape((5,2))
T = X[:,0:1] + 2 * X[:,1:2] + np.random.uniform(-1, 1,(5, 1))
print('Inputs')
print(X)
print('Targets')
print(T)


# In[ ]:

model = train(X, T)
model


# In[ ]:

predicted = use(model, X)
predicted


# In[ ]:

rmse(predicted, T)


# In[ ]:

modelSGD = trainSGD(X, T, 0.01, 100)
modelSGD


# In[ ]:

predicted = use(modelSGD, X)
predicted


# In[ ]:

rmse(predicted, T)


# ## Data

# Download ```energydata_complete.csv``` from the [Appliances energy prediction Data Set ](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) at the UCI Machine Learning Repository. Ignore the first column (date and time), use the next two columns as target variables, and use all but the last two columns (named rv1 and rv2) as input variables. 
# 
# There are 19735 total data instances, Appliances and lights are the variables we want to predict. The other variables are our training variables. The data was collected by Luis Candanedo with a ZigBee wireless sensor network.

# In[261]:

file = np.genfromtxt('energydata_complete.csv', dtype='str',delimiter=',',deletechars='"')
file = np.char.replace(file, '"', '')
file = np.char.replace(file, ' ', '')
file = np.delete(file, 0,1)
file = np.delete(file, -1,1)
file = np.delete(file, -1,1)
names = file[0]
names = names.astype(np.str)
data = file[1:]
data = data.astype(np.float)

Tenergy = np.take(data,[0,1],1)
Xenergy = np.take(data,range(2,26),1)
Tnames = np.take(names,[0,1])
Xnames = np.take(names,range(2,26))


# In[264]:

plt.figure(figsize=(10,10))
plt.subplot(2, 1, p+1)
plt.plot(Xenergy, Tenergy, 'o')
plt.xlabel("Predictor Variables")
plt.ylabel("Target Variables" )


# ## Results

# Apply your functions to the data.  Compare the error you get as a result of both training functions.  Experiment with different learning rates for ```trainSGD``` and discuss the errors.
# 
# Make some plots of the predicted energy uses and the actual energy uses versus the sample index.  Also plot predicted energy use versus actual energy use.  Show the above plots for the appliances energy use and repeat them for the lights energy use. Discuss your observations of each graph.
# 
# Show the values of the resulting weights and discuss which ones might be least relevant for fitting your linear model.  Remove them, fit the linear model again, plot the results, and discuss what you see.

# #### Training with regular Train

# In[198]:

trained = train(Xenergy,Tenergy)
used = use(trained, Xenergy)
rmse1 = rmse(used,Tenergy)
rmse1


# #### Training with SGD LR 0.01

# In[ ]:

trainedSGD = trainSGD(Xenergy,Tenergy, .01, 10)
usedSGD = use(trainedSGD, Xenergy)
rmse2 = rmse(usedSGD,Tenergy)
rmse2


# ^This is bad, as we can see later on, our predictions are way off. Let's try it with a smaller learning rate!

# #### Training with SGD LR 0.001

# In[ ]:

trainedSGD3 = trainSGD(Xenergy,Tenergy, .001, 100)
usedSGD3 = use(trainedSGD2, Xenergy)
rmse3 = rmse(usedSGD3,Tenergy)
rmse3


# ^Getting closer! Let's go even smaller

# #### Training with SGD LR 0.00001

# In[ ]:

trainedSGD4 = trainSGD(Xenergy,Tenergy, .00001, 100)
usedSGD4 = use(trainedSGD4, Xenergy)
rmse4 = rmse(usedSGD4,Tenergy)
rmse4


# ^This is pretty good! and smaller and we will get a less accurate prediction. We'll stick with this one

# # Plotting a sample index from 1-100 (train and best RMSE of trainSGD):

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    for i in range(100):
        plt.subplot(2, 1, p+1)
        plt.plot(i, Tenergy[i, p], 'or',)
        plt.plot(i, used[i, p], 'ob',)
        plt.xlabel("index")
        plt.ylabel("usage" )


# ^It looks like our prediction for the first column of data was actually pretty good. It seems like a few outliers might have thrown off our results. However, the general curve of the line is fairly similar. The second column is less accurate. I would think it is because the amount of zeros from indexes 50-80 might have thrown off our predictions

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    for i in range(100):
        plt.subplot(2, 1, p+1)
        plt.plot(i, Tenergy[i, p], 'or',)
        plt.plot(i, usedSGD4[i, p], 'ob',)
        plt.xlabel("index")
        plt.ylabel("usage" )


# ^these plots appear fairly similar to our regular train plots

# # Plotting predicted energy use vs. actual

# #### plotting regular train

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    plt.subplot(2, 1, p+1)
    plt.plot(used[:, p], Tenergy[:, p], 'o')
    plt.xlabel("Predicted ")
    plt.ylabel("Actual " )
    a = max(min(used[:, p]), min(Tenergy[:, p]))
    b = min(max(used[:, p]), max(Tenergy[:, p]))
    plt.plot([a, b], [a, b], 'r', linewidth=3)


# #### Plotting LR .01

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    plt.subplot(2, 1, p+1)
    plt.plot(usedSGD[:, p], Tenergy[:, p], 'o')
    plt.xlabel("Predicted ")
    plt.ylabel("Actual " )
    a = max(min(usedSGD[:, p]), min(Tenergy[:, p]))
    b = min(max(usedSGD[:, p]), max(Tenergy[:, p]))
    plt.plot([a, b], [a, b], 'r', linewidth=3)


# #### Plotting LR .001

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    plt.subplot(2, 1, p+1)
    plt.plot(usedSGD2[:, p], Tenergy[:, p], 'o')
    plt.xlabel("Predicted ")
    plt.ylabel("Actual " )
    a = max(min(usedSGD2[:, p]), min(Tenergy[:, p]))
    b = min(max(usedSGD2[:, p]), max(Tenergy[:, p]))
    plt.plot([a, b], [a, b], 'r', linewidth=3)


# #### Plotting LR .001

# In[ ]:

plt.figure(figsize=(10,10))
for p in range(2):
    plt.subplot(2, 1, p+1)
    plt.plot(usedSGD3[:, p], Tenergy[:, p], 'o')
    plt.xlabel("Predicted ")
    plt.ylabel("Actual " )
    a = max(min(usedSGD3[:, p]), min(Tenergy[:, p]))
    b = min(max(usedSGD3[:, p]), max(Tenergy[:, p]))
    plt.plot([a, b], [a, b], 'r', linewidth=3)


# In[231]:

print(trained["w"])
print(trainedSGD4["w"])


# ^We can see that some of these have a very low weight! Let's try pruning some to see if we can do better

# Unfortunately I was unable to figure out how to remove these weights from w, and get it to still work with the use function. I realize that if we were to remove the low weights from our array, we would get a better prediction

# In[ ]:

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



