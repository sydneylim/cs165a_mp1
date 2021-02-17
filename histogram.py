import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('covid_train.csv')
plt.hist(data, normed=True, bins='auto')