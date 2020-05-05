import math
import numpy as np

def random_mini_batches(X, Y, mini_batch_size=128, seed=0 ):
    
    np.random.seed(seed)  ######THIS IS DONE SO AS TO CREATE DIFFERENT BATCHES AFTEER EACH "epoch"
    m = X.shape[1]  #### Number of training examples
    mini_batches = []
    
    #Step 1: Shuffle X, Y
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(10,m) #10 for 10 classes and also that Y has been one hot encoded
    
    #Step 2: Create Mini Batches
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m% mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k+1)*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (k+1)*mini_batch_size:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches


# In[ ]:




