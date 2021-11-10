import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:13:19 2021

@author: stefan
"""

from neural_gas import neural_gas
from generator import Generator

g = Generator(obs_num=50)

means = [[2,1],
        [20, 17],
        [8, 9]]
sigmas = [2, 2, 2]
data = g.gen(means, sigmas)

NG = neural_gas(3, 2, 0.1, 0.5)
NG.setup(data)
NG.train(data, 100)