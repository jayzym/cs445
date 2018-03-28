
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

# # Assignment 4: Classification with QDA, LDA, and Logistic Regression

# *Type your name here and rewrite all of the following sections.  Add more sections to present your code, results, and discussions.*

# ## Overview

# In this assignment, you will make a new version of your ```NeuralNetwork``` class called ```NeuralNetworkClassifier```. You will then apply ```QDA```, ```LDA``` and your ```NeuralNetworkClassifier``` to a classification problem and discuss the results.  The ```tanh``` function will be used as the activation function for ```NeuralNetworkClassifier```.

# ### NeuralNetworkClassifier

# Copy your ```neuralnetworksA2.py``` into a new file named ```neuralnetworksA4.py```.  Define a new class named ```NeuralNetworkClassifier``` that extends ```NeuralNetwork```.  The following code cell indicates which methods you must override, with comments instructing you what you must do to complete it.  Add this class to your ```neuralnetworksA4.py``` file.

# ### Test It

# In[ ]:

import numpy as np
import neuralnetworksA4 as nn


# In[ ]:

X = np.arange(10).reshape((-1, 1))
T = np.array([1]*5 + [2]*5).reshape((-1, 1))


# In[ ]:

netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(X, T, 20)
print(netc)
print('T, Predicted')
print(np.hstack((T, netc.use(X))))


# ## Addition to ```partition``` function

# Add the keyword parameter ```classification``` with a default value of ```False``` to your ```partition``` function.  When its value is set to ```True``` your ```partition``` function must perform a stratified partitioning as illustrated in lecture notes [12 Introduction to Classification](http://nbviewer.ipython.org/url/www.cs.colostate.edu/~anderson/cs445/notebooks/12%20Introduction%20to%20Classification.ipynb)     

# In[ ]:

import mlutilities as ml
Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.6, classification=True)


# In[ ]:

Xtrain


# In[ ]:

Ttrain


# In[ ]:

Xtest


# In[ ]:

Ttest


# ### Example with toy data

# Use the above data to compare QDA, LDA, and linear and nonlinear logistic regression.

# In[ ]:

import qdalda
qda = qdalda.QDA()
qda.train(Xtrain, Ttrain)
Ytrain = qda.use(Xtrain)
Ytest = qda.use(Xtest)


# In[ ]:

print(np.hstack((Ttrain, Ytrain)))


# In[ ]:

np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[ ]:

print(np.hstack((Ttest, Ytest)))


# In[ ]:

np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[ ]:

lda = qdalda.LDA()
lda.train(Xtrain, Ttrain)
Ytrain = lda.use(Xtrain)
Ytest = lda.use(Xtest)


# In[ ]:

print(np.hstack((Ttrain, Ytrain)))


# In[ ]:

print(np.hstack((Ttest, Ytest)))


# In[ ]:

np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[ ]:

np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[ ]:

ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[ ]:

ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# In[ ]:

netc = nn.NeuralNetworkClassifier(X.shape[1], [5, 5], len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[ ]:

print(np.hstack((Ttrain, Ytrain)))


# In[ ]:

print(np.hstack((Ttest, Ytest)))


# In[ ]:

np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[ ]:

np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[ ]:

ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[ ]:

ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# Remember that linear logistic regression can be applied by specifying 0 hidden units.

# In[ ]:

netc = nn.NeuralNetworkClassifier(X.shape[1], 0, len(np.unique(T)))
netc.train(Xtrain, Ttrain, 100)
print(netc)
print('T, Predicted')
Ytrain = netc.use(Xtrain)
Ytest = netc.use(Xtest)


# In[ ]:

ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[ ]:

ml.confusionMatrix(Ttest, Ytest, [1, 2]);


# ## Apply to data from orthopedic patients

# Download ```column_3C_weka.csv``` from [this Kaggle site](https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients).  Use the column named ```class``` to create your target class labels. Apply QDA, LDA, linear logistic regression, and nonlinear logistic regression to this data.  Experiment with different hidden layer structures and numbers of iterations and describe what you find.
# 
# Partition data into 80% for training and 20% for testing, with ```shuffle=True```.
# 
# Print percents of training and testing samples correctly classified by QDA, LDA and various neural network classifiers.  Also print confusion matrices for training and for testing samples for each classifier.  Discuss the relative performance of your classifiers.

# In[54]:

file = np.genfromtxt('column_3C_weka.csv', dtype='str',delimiter=',',deletechars='"')
file = np.char.replace(file, '"', '')
file = np.char.replace(file, ' ', '')
file = np.delete(file,0,0)


# In[55]:

PT = np.take(file,[-1],1)
PT = np.char.replace(PT, 'Hernia', '0')
PT = np.char.replace(PT, 'Spondylolisthesis', '1')
PT = np.char.replace(PT, 'Normal', '2')
file = np.char.replace(file, ' ', '')
PX = np.take(file,[0,1,2,3,4,5],1)
PT = PT.astype(np.float)
PX = PX.astype(np.float)


# In[56]:

PT


# In[57]:

PX


# In[58]:

Xtrain, Ttrain, Xtest, Ttest = ml.partition(PX, PT, 0.8, classification=True, shuffle=True)


# In[59]:

nne = nn.NeuralNetworkClassifier(PX.shape[1], 0, len(np.unique(PT)))


# In[60]:

nne.train(Xtrain, Ttrain, 100)
print(nne)
print('T, Predicted')
print(np.hstack((PT, nne.use(PX))))
Ytrain = nne.use(Xtrain)
Ytest = nne.use(Xtest)


# In[61]:

np.sum(Ttrain == Ytrain) / len(Ttrain) * 100


# In[62]:

ml.confusionMatrix(Ttrain, Ytrain, [1, 2]);


# In[52]:

np.sum(Ttest == Ytest) / len(Ttest) * 100


# In[53]:

lda = qdalda.LDA()
lda.train(Xtrain, Ttrain)
Ytrain = lda.use(Xtrain)
Ytest = lda.use(Xtest)


# ## Grading and Check-in

# In[ ]:




# Your notebook will be run and graded automatically. Test this grading process by first downloading [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs445/notebooks/A4grader.tar) and extract A4grader.py from it.   

# In[ ]:

X = np.vstack((np.arange(20), [7, 4, 5, 5, 8, 4, 6, 7, 4, 9, 4, 2, 6, 6, 3, 3, 7, 2, 6, 4])).T
T = np.array([1]*8 + [2]*8 + [3]*4).reshape((-1, 1))


# In[ ]:

Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, 0.8, classification=True, shuffle=False)


# In[ ]:

get_ipython().magic('run -i A4grader.py')


# # Extra Credit

# Earn 1 extra credit point by doing a few experiments with different neural network classifiers using the ReLU activation function on the orthopedic data. Discuss any differences you see from your earlier results that used tanh.

# In[ ]:



