import pandas as pd
import numpy as np

class SelfAttention:
    # init에서 Q, K, V를 받기?
    def __init__(self, Q, K, V):
        self.softmax(Q@K.T)@V
    
    def softmax(X):
        return np.exp(X) / np.sum(np.exp(X))