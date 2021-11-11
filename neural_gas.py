import numpy as np

from numpy import exp
from numpy.random import uniform,choice,randint
from numpy.linalg import norm

from matplotlib import pyplot as plt

class NeuralGas:
    
    def __init__(self,n_nodes,feature_dim,step_size,neighbour_hood_range):
        
        self._N = n_nodes
        self._epsilon = step_size
        self._nhr = neighbour_hood_range
        self._f_dim = feature_dim
        self._W = np.zeros((self._N,self._f_dim))
        
        
    def setup(self,X):
        # get volume filled by data
        volume_min = X.min(axis=0)
        volume_max = X.max(axis=0)
        # uniformly sample N nodes
        # for each dimension of the data
        for i,(mmin,mmax) in enumerate(list(zip(volume_min,volume_max))):
            # sample uniformly in this dimension
            self._W[:,i] = uniform(low=mmin,high=mmax,size=(1,self._N))
            
    def train(self,X,n_iterations,animate=True):
        # get number of samples
        m = X.shape[0]
        # create figure for plot if wanted
        # while there is still training steps left:
        for t in range(n_iterations):
            # draw a value from X that I call x
            sample_i = randint(0,m)
            x = X[sample_i,:]
            # calculate the distance of all feature vectors to x
            dist = np.zeros((self._N,2))
            for i,w in enumerate(self._W):
                # calculate distance to x
                dist[i,0] = i
                dist[i,1] = norm(w-x)
            # sort feature vectors in ascending order of distance to x
            dist = dist[np.argsort(dist[:, 1])]
            # from closest to farthest update the vector according to
            for k,d in enumerate(dist):
                # select the right feature vector
                w_idx = int(d[0])
                w = self._W[w_idx,:]
                # learning rule
                self._W[w_idx,:] = w+self._epsilon*exp(-k/self._nhr)*(x-w) 
            if animate:
                plt.clf()
                plt.scatter(X[:,0],X[:,1])
                plt.scatter(self._W[:,0],self._W[:,1])
                # plot lines
                for idx,w0 in enumerate(self._W):
                    for _,w1 in enumerate(self._W[idx+1:]):
                        x_pos = [w0[0],w1[0]]
                        y_pos = [w0[1],w1[1]]
                        plt.plot(x_pos, y_pos, 'black', linestyle='--', marker='')
                plt.draw()
                plt.pause(0.25)
                
                