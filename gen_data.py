import numpy as np
import pandas as pd


a_dat = np.linspace(0,100,1000)
a_dat = a_dat + np.random.randn(len(a_dat))

b_dat = np.linspace(0,100,1000)
b_dat = b_dat + np.random.randn(len(b_dat))

y_true = a_dat*a_dat + b_dat*b_dat +2*a_dat*b_dat

print(y_true)
