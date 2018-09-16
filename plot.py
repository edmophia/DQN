# libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from hyperparameters import *
sns.set()
%matplotlib inline

def mov_avg(array, window_size = 1):
    window_size = 100
    subarrays = [array[i:i+window_size] for i in range(array.size-window_size)]
    return np.mean(subarrays,axis=1)

def make_plot():
    i = 4
    # load data
    pkl_file = open(RESULT_PATH,'rb')
    result = pickle.load(pkl_file)
    pkl_file.close() 

    # use the plot function
    plt.figure(figsize=(10,10))
    plt.plot(range(len(mov_avg(result,10))),mov_avg(result,10),label='DQN',color='red',alpha=0.7)
    plt.legend()
    plt.xlabel('sim steps(10k)')
    plt.ylabel('Return')
    FIG_PATH = './data/plots/result.png'
    plt.savefig(FIG_PATH)
    plt.show()

make_plot()