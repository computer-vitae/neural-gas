import numpy as np
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, features=2, obs_num=10) -> None:
        self.features = features
        self.obs_num = 10

    def set_shape(self, shape):
        self.shape=shape

    def gen(self, means, sigmas):
        data = np.zeros((len(means)*self.obs_num, self.features), dtype=float)

        if  isinstance(means, (list, tuple, np.array)):
            for i, (mu, s) in enumerate(zip(means, sigmas)):
                d = np.random.normal(mu, s, (self.obs_num, self.features))
                data[i*self.obs_num:(i+1)*self.obs_num, :] = d
        return data


# g = Generator(obs_num=50)

# means = [[2,1],
#         [20, 17],
#         [8, 9]]
# sigmas = [2, 2, 2]
# data = g.gen(means, sigmas)

# plt.scatter(data[:, 0], data[:, 1])
# plt.show()
