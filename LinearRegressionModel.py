# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:42:09 2021

@author: akif-
"""

import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0

plt.style.use('seaborn')


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Normal equation formula to update theta parameters.

        Args:
            Training examples for X and y
        """
        
        self.theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y)) # Equation for normal function to calculate the best paramater theta


    def fit_GD(self, X, y):
        """gradient descent algorithm.

        """
        # Declare learning rate and initial iteration value
        L_rate = 0.01
        k = 0
        n,m = np.shape(X)
        self.theta=np.zeros(m) # Set inital thetas to zero and the size to # of features
        iterations = 10000
        while k < iterations: # Iterate over all iterations          
            for c in range(0, m): # For each theta value in the training set (size = features)
                pred_y = np.dot(X,self.theta) # Calculate predicted y value using h theta of x function
                self.theta[c] = self.theta[c] - L_rate*(sum((pred_y-y)*X[:,c])) #update theta accordingly
                k+=1

        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """stochastic gradient descent algorithm.

        """
        # *** SAME THING AS GD except this time we are not calculating the sum of all values 0 to n in the loop
        # We created an outer loop for that sum instead
        L_rate = 0.01
        k = 0
        n,m = np.shape(X)
        print (n)
        self.theta=np.zeros(m)
        iterations = 10000
        while k < iterations:  
            for i in range (0,n):
                for j in range(0, m):
                    pred_y = np.dot(X[i],self.theta)
                    self.theta[j] = self.theta[j] - L_rate*((pred_y-y[i])*X[i,j])
                    k+=1
        print (self.theta)

        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        output the new matrix X with polynomial features up to exponent k
     
        """
        # *** START CODE HERE **
        newArray = []
        m = np.shape(X) # Get the # of features of training example input
        for i in range(0,k+1): # For loop to iterate through how many more feature vectors we have to add
            newArray.append(np.power(X[:,1],i))   # Add the polynomial features
        newArray = np.transpose(newArray) # Transpose the new features
        X = np.stack(newArray) #Stack it on top of the array
        return X
        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        add cosine function to the polynomial matrix

        """
        newArray = []
        m = np.shape(X)
        
        for i in range(0,k+1):
            newArray.append(np.power(X[:,1],i))
        # Now after all the k polynomials are added, add a cos(x) to the matrix too   
        newArray.append(np.cos(X[:,1]))     
        newArray = np.transpose(newArray)
        X = np.stack(newArray)
        return X
        # *** END CODE HERE ***

    def predict(self, X):
        """
        The h theta of x function for creating predictions of a hypothetical point X with parameters theta

        """
        hThetaofX = []
        np.array(hThetaofX)
        hThetaofX = np.dot(X,self.theta) # The function for calculating h theta of X
        return hThetaofX


def run_exp(train_path, cosine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.pdf'):

    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    lm = LinearModel()
    """
    RUNNING ALL THE EXPERIMENTS
  """
    result1 = lm.create_poly(3, train_x)
    lm.fit(result1,train_y)
    newPlot = lm.create_poly(3, plot_x)
    plot_y = lm.predict(newPlot)
    
    result2 = lm.create_poly(3, train_x)
 #   lm.fit_GD(result2, train_y)
  #  newPlot2= lm.create_poly(3, plot_x)
   # plot_y2 = lm.predict(newPlot2)
    
    result3 = lm.create_poly(3, train_x)
#   lm.fit_SGD(result3, train_y)
 #   newPlot3 = lm.create_poly(3,plot_x)
  #  plot_y3 = lm.predict(newPlot3)
  #  print(newPlot)
  #  lm.fit_GD(train_x, train_y)
   # util.plot(newPlot, plot_y, lm.theta, save_path)
#    result1 = lm.create_poly(3, train_x)
#    lm.fit(result1,train_y)
 #   newPlot = lm.create_poly(3, plot_x)
  #  plot_y = lm.predict(newPlot)
    
    result2 = lm.create_poly(5, train_x)
    lm.fit(result2,train_y)
    newPlot2 = lm.create_poly(5, plot_x)
    plot_y2 = lm.predict(newPlot2)
    
    result3 = lm.create_poly(10, train_x)
    lm.fit(result3,train_y)
    newPlot3 = lm.create_poly(10, plot_x)
    plot_y3 = lm.predict(newPlot3)
    
    result4 = lm.create_poly(20, train_x)
    lm.fit(result4,train_y)
    newPlot4 = lm.create_poly(20, plot_x)
    plot_y4 = lm.predict(newPlot4)
    
    cos1 = lm.create_cosine(3, train_x)
    lm.fit(cos1, train_y)
    cosPlot = lm.create_cosine(3, plot_x)
    plotCos = lm.predict(cosPlot)
    
    cos2 = lm.create_cosine(5, train_x)
    lm.fit(cos2, train_y)
    cosPlot2 = lm.create_cosine(5, plot_x)
    plotCos2 = lm.predict(cosPlot2)
    
    cos3 = lm.create_cosine(10, train_x)
    lm.fit(cos3, train_y)
    cosPlot3 = lm.create_cosine(10, plot_x)
    plotCos3 = lm.predict(cosPlot3)
    
    cos4 = lm.create_cosine(20, train_x)
    lm.fit(cos4, train_y)
    cosPlot4 = lm.create_cosine(20, plot_x)
    plotCos4 = lm.predict(cosPlot4)
    for k in ks:
        
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***


        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2.5, 2.5)
     #   plt.plot(plot_x, plot_y,label='k={:d}, fit={:s}'.format(k, f_type))

    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    #plt.plot(train_x,train_y)
    plt.scatter(train_x[:, 1], train_y)
   # print(plot_y2)
    plt.ylim(-2.5, 2.5)
#    plt.plot(plot_x, plotCos, label='k={:d}, fit={:s}'.format(k, 'k=3')) #first green
 #   plt.plot(plot_x, plotCos2) #purple
  #  plt.plot(plot_x, plotCos3) #blue
   # plt.plot(plot_x, plotCos4) #green
    
    plt.plot(plot_x, plot_y) 
   # plt.plot(plot_x, plot_y2, label='k={:d}, fit={:s}'.format(k, 'k=3')) #first green
  #  plt.plot(plot_x, plot_y3)
   # plt.plot(plot_x, plot_y4) #purple
   # plt.plot(plot_x, plot_y3) #blue
   # plt.plot(plot_x, plot_y4) #green
    
    



def main(medium_path, small_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # = plt.read_csv('medium.csv')
    run_exp(medium_path)
    
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
