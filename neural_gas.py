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



class GrowingNeuralGas:
        
        def __init__(self, n_nodes, feature_dim, step_size, max_age=50):
        
            self._N = n_nodes
            self._epsilon = step_size
            self._f_dim = feature_dim
            self._W = np.zeros((self._N,self._f_dim))
            self.AM = np.random.choice((0, 1), (n_nodes, n_nodes))
            self.age = np.zeros((n_nodes, n_nodes))
            self.epsilon_a = 0.2
            self.epsilon_n = 0.006
            self.delta_error = np.zeros(n_nodes)
            self.max_age = max_age
            self.input_counter = 0
            self.l = 100
            self.d = 0.995

        def setup(self,X):
            # get volume filled by data
            volume_min = X.min(axis=0)
            volume_max = X.max(axis=0)
            # uniformly sample N nodes
            # for each dimension of the data
            for i,(mmin,mmax) in enumerate(list(zip(volume_min,volume_max))):
                # sample uniformly in this dimension
                self._W[:,i] = uniform(low=mmin,high=mmax,size=(1,self._N))

        def get_adjacent(self, i):
            return np.nonzero(self.AM[i])

        def k_dist(self, x):
            '''
            Finds the ditance from x to all the centers
            
            '''
            #TODO:In the future we could do this useing scipy for faster computation 
            dist = np.zeros((self._N,2))
            for i,w in enumerate(self._W):
                # calculate distance to x
                dist[i,0] = i
                dist[i,1] = norm(w-x)
            # sort feature vectors in ascending order of distance to x
            dist = dist[np.argsort(dist[:, 1])]
            return dist

        def update_nodes(self, x):
            dists = self.k_dist(x)
            # get the 2 closest nodes
            s1 = dists[0][0]
            s2 = dists[1][0]
            # log the age of the vectors
            self.increment_age(s1, x)
            # update the nearest node
            self._W[s1] = self._W[s1] + self.epsilon_a*(x - self._W[s1])
            # get the adjacent nodes and update them
            adj = self.get_adjacent(s1)
            for i in adj:
                self._W[i] = self._W[i] + self.epsilon_n*(x - self._W[i])
            
            self.update_edges(s1, s2)

        def add_node(self, ):
            if not (self.input_counter % self.l):
                pass

        def update_edges(self, s1, s2):
            if self.AM[s1, s2]:
                # zero the age of the edge if it exists
                self.age[s1, s2] = 0
            else:
                # create the edge if it does not exist
                self.AM[s1, s2] = 1
            #Remove edges that are older than the max
            indices = np.argwhere(self.age > self.max_age)
            for i in indices:
                self.AM[i[0], i[1]] = 0
                self.AM[i[1], i[0]] = 0
            #TODO: check for verteces with no edges to remove as well
        
        def increment_age(self, s1, x):
            #indices of the adjacent vectors
            adj = np.nonzero(self.AM[s1])
            #inrease the age of the adjacent vector edges
            for i, j in zip(*adj):
                self.age[i, j] += 1
            # add squared distance to the local error counter
            self.delta_error[s1] += np.abs(self._W[s1] - x)**2

                