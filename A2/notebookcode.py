
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

# # Assignment 2: Neural Network Regression

# *Type your name here and rewrite all of the following sections.  Add more sections to present your code, results, and discussions.*

# ## Overview

# The goal of this assignment is to learn about object-oriented programming in python and to gain some experience in comparing different sized neural networks when applied to a data set.
# 
# Starting with the ```NeuralNetwork``` class from the lecture notes, you will create one new version of that class, apply it to a data set, and discuss the results.

# ## Required Code

# Start with the ```NeuralNetwork``` class defined in lecture notes 09. Put that class definition as written into *neuralnetworks.py* into your current directory.  Also place *mlutilities.py* from lecture notes 09 in your current directory. If this is done correctly, then the following code should run and produce results similar to what is shown here.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

import neuralnetworks as nn

X = np.arange(10).reshape((-1,1))
T = np.sin(X)

nnet = nn.NeuralNetwork(1, [10], 1)
nnet.train(X, T, 100, verbose=True)
nnet


# In[3]:

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(nnet.getErrors())

plt.subplot(3, 1, 2)
plt.plot(X, T, 'o-', label='Actual')
plt.plot(X, nnet.use(X), 'o-', label='Predicted')

plt.subplot(3, 1, 3)
nnet.draw()


# Now let's extract the parts of the neural network code that refer to the activation function and its derivative into two new methods.  Copy your file *neuralnetworks.py* into a new file named *neuralnetworksA2.py*.  Modify the code in *neuralnetworksA2.py* by adding these two methods to the ```NeuralNetwork``` class:
# 
#     def activation(self, weighted_sum):
#         return np.tanh(weighted_sum)
#         
#     def activationDerivative(self, activation_value):
#         return 1 - activation_value * activation_value
#         
# Now replace the code in the appropriate places in the ```NeuralNetwork``` class so that ```np.tanh``` is replaced with a call to the ```self.activation``` method and its derivative is replaced by calls to ```self.activationDerivative```.

# In[4]:

import neuralnetworksA2 as nn2

nnet = nn2.NeuralNetwork(1, [10], 1)


# In[5]:

[nnet.activation(s) for s in [-2, -0.5, 0, 0.5, 2]]


# In[6]:

[nnet.activationDerivative(nnet.activation(s)) for s in [-2, -0.5, 0, 0.5, 2]]


# In[7]:

nnet.train(X, T, 100, verbose=True)
nnet


# In[8]:

plt.figure(figsize=(8, 12))
plt.subplot(3, 1, 1)
plt.plot(nnet.getErrors())

plt.subplot(3, 1, 2)
plt.plot(X, T, 'o-', label='Actual')
plt.plot(X, nnet.use(X), 'o-', label='Predicted')

plt.subplot(3, 1, 3)
nnet.draw()


# ## Neural Network Performance with Different Hidden Layer Structures and Numbers of Training Iterations

# ### Example with Toy Data

# Using your new ```NeuralNetwork``` class, you can compare the error obtained on a given data set by looping over various hidden layer structures.  Here is an example using the simple toy data from above.

# In[9]:

import random

nRows = X.shape[0]
rows = np.arange(nRows)
np.random.shuffle(rows)
nTrain = int(nRows * 0.8)
trainRows = rows[:nTrain]
testRows = rows[nTrain:]
Xtrain, Ttrain = X[trainRows, :], T[trainRows, :]
Xtest, Ttest = X[testRows, :], T[testRows, :]


# In[10]:

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[11]:

def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))


# In[12]:

import pandas as pd

errors = []
hiddens = [0] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
print('hiddens =', hiddens)
for hids in hiddens: 
    nnet = nn.NeuralNetwork(Xtrain.shape[1], hids, Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, 500)
    errors.append([hids, rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])
errors = pd.DataFrame(errors)
print(errors)

plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), hiddens, rotation=30, horizontalalignment='right')
plt.grid(True)


# For this data (and the random shuffling of the data used when this notebook was run), `[10, 10, 10, 10]` produced the lowest test error.  
# 
# Now, using the best hidden layer structure found, write the code that varies the number of training iterations. The results you get will be different from the ones shown below.

# In[13]:

errors = []
nIterationsList = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

for n in nIterationsList:
    nnet = nn.NeuralNetwork(Xtrain.shape[1], [50], Ttrain.shape[1])
    nnet.train(Xtrain, Ttrain, n)
    errors.append([[50], rmse(Ttrain, nnet.use(Xtrain)), rmse(Ttest, nnet.use(Xtest))])
errors = pd.DataFrame(errors)
print(nIterationsList)
print(errors)
plt.figure(figsize=(10, 10))
plt.plot(errors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(errors.shape[0]), nIterationsList) # , rotation=30, horizontalalignment='right')
plt.grid(True)


# ### Experiments wtih Automobile Data

# Now, repeat the above steps with the automobile mpg data we have used before.  Set it up to use 
# 
#   * cylinders,
#   * displacement,
#   * weight,
#   * acceleration,
#   * year, and
#   * origin
#   
# as input variables, and
# 
#   * mpg
#   * horsepower
#   
# as output variables.

# In[14]:

def makeMPGData(filename='auto-mpg.data'):
    def missingIsNan(s):
        return np.nan if s == b'?' else float(s)
    data = np.loadtxt(filename, usecols=range(8), converters={3: missingIsNan})
    print("Read",data.shape[0],"rows and",data.shape[1],"columns from",filename)
    goodRowsMask = np.isnan(data).sum(axis=1) == 0
    data = data[goodRowsMask,:]
    print("After removing rows containing question marks, data has",data.shape[0],"rows and",data.shape[1],"columns.")
    X = data[:,1:]
    X = np.delete(X,2,axis=1)
    T = data[:,0:1]
    T = np.append(T,data[:,3:4],axis=1)
    Xnames =  ['bias', 'cylinders','displacement','weight','acceleration','year','origin']
    Tname = ['mpg','horsepower']
    return X,T,Xnames,Tname


# In[15]:

Xa,Ta,Xanames,Taname = makeMPGData()


# In[16]:

naRows = Xa.shape[0]
arows = np.arange(naRows)
np.random.shuffle(arows)
naTrain = int(naRows * 0.8)
atrainRows = arows[:naTrain]
atestRows = arows[naTrain:]
Xatrain, Tatrain = Xa[atrainRows, :], Ta[atrainRows, :]
Xatest, Tatest = Xa[atestRows, :], Ta[atestRows, :]


# In[17]:

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[18]:

aerrors = []
ahiddens = [0] + [[nu] * nl for nu in [1, 5, 10, 20, 50] for nl in [1, 2, 3, 4, 5]]
print('hiddens =', ahiddens)
for hids in ahiddens: 
    annet = nn.NeuralNetwork(Xatrain.shape[1], hids, Tatrain.shape[1])
    annet.train(Xatrain, Tatrain, 500)
    aerrors.append([hids, rmse(Tatrain, annet.use(Xatrain)), rmse(Tatest, annet.use(Xatest))])
aerrors = pd.DataFrame(aerrors)
print(aerrors)

plt.figure(figsize=(10, 10))
plt.plot(aerrors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(aerrors.shape[0]), ahiddens, rotation=30, horizontalalignment='right')
plt.grid(True)


# ^After running this several times, it appears that the structures that most often show up as the best (lowest RMSE) are [5,5] and [50,50]. [50] showed up sometimes. Overal it seemed to vary a bit. I'll choose [5,5] as my structure

# In[ ]:

aerrors = []
anIterationsList = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

for n in anIterationsList:
    annet = nn.NeuralNetwork(Xatrain.shape[1], [5,5], Tatrain.shape[1])
    annet.train(Xatrain, Tatrain, n)
    aerrors.append([[5,5], rmse(Tatrain, annet.use(Xatrain)), rmse(Tatest, annet.use(Xatest))])
aerrors = pd.DataFrame(aerrors)
print(anIterationsList)
print(aerrors)
plt.figure(figsize=(10, 10))
plt.plot(aerrors.values[:, 1:], 'o-')
plt.legend(('Train RMSE', 'Test RMSE'))
plt.xticks(range(aerrors.shape[0]), anIterationsList) # , rotation=30, horizontalalignment='right')
plt.grid(True)


# ^After determining my best hidden structure to be [5,5]. I ran the iterations several times. While the results were all over the place. Most often the best RMSE was greater than 200. The graph will usually fluctuate after 200, and then settle down and get closer to zero as the number of iterations approached 500. Sometimes 500 would even be the best number of iterations. Overall I think it's fair to say that a higher number of iterations tends to result in a better score.

# ## Grading and Check-in

# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect execution score of  60 / 60 if your functions and class are defined correctly. The remaining 40 points will be based on the results you obtain from the comparisons of hidden layer structures and numbers of training iterations on the automobile data.
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.  Your working director must also contain `neuralnetworksA2.py` and `mlutilities.py` from lecture notes.
# 
# Combine your notebook and `neuralnetworkA2.py` into one zip file or tar file.  Name your tar file `Lastname-A2.tar` or your zip file `Lastname-A2.zip`.  Check in your tar or zip file using the `Assignment 2` link in Canvas.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include other tests.

# In[ ]:

get_ipython().magic('run -i A2grader.py')

