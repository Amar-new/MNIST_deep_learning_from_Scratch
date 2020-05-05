import pandas as pd
import numpy as np

def load_train_test_dataset(platform_name):
    '''################# JUPYTER RETRIEVING PROCESS #############'''
    if platform_name=="Jupyter":
        train=pd.read_csv("Dataset/mnist_train.csv")      #shape(60000x785) -- one column is the label column
        test=pd.read_csv("Dataset/mnist_test.csv")        #shape(10000x785)
        
        print("-------------Train Dataframe:---------\n\n", train.head())
        print("\n\n--------------Test Dataframe:---------\n\n", test.head(),"\n\n")

        #Create X_train and Y_train
        #X_train--(60000,784)    Y_train--(60000,)
        #Y_train ---- need to be developed to a row
        X_train, Y_train= train.iloc[:,1:785], train.iloc[:,0]
        print("\nDataframe X_train shape",X_train.shape)
        print("Dataframe Y_train shape",Y_train.shape)

        #Create X_test and Y_test
        #X_test--(10000,784)    Y_test--(10000,)
        #Y_test ---- need to be developed to a row
        X_test, Y_test= test.iloc[:,1:785], test.iloc[:,0]
        print("\nDataframe X_test shape",X_test.shape)
        print("Dataframe Y_test shape",Y_test.shape)

        #Convert Train Dataframe to Numpy 
        X_train=X_train.to_numpy()
        Y_train=Y_train.to_numpy()
        #Reshape X_train & Y_train
        X_train=X_train.transpose()
        Y_train=Y_train.transpose().reshape((1,60000))
        print("\nNumpy array X_train shape: ",X_train.shape)
        print("NUmpy array Y_train shape: ",Y_train.shape)

        #Convert Test Dataframe to Numpy
        X_test=X_test.to_numpy()
        Y_test=Y_test.to_numpy()
        #Reshape X_test & Y_test
        X_test=X_test.transpose()
        Y_test=Y_test.transpose().reshape((1,10000))
        print("\nNumpy array X_test shape: ",X_test.shape)
        print("NUmpy array Y_test shape: ",Y_test.shape)
    
        return X_train, Y_train, X_test, Y_test
    
    '''################## COLAB DATA RETRIEVING PROCESS #######################'''
    if platform_name=="Colab":
        
        ''' ######### UNZIP .gz FILES #########'''
        
        '''########### READ .gz FILES #############'''
        '''########### READ THE TRAIN IMAGE FILE #############'''
        import gzip
        f1 = gzip.open('train-images-idx3-ubyte.gz','r')

        image_size = 28
        num_images_train = 60000

        f1.read(16)
        buf = f1.read(image_size * image_size * num_images_train)
        X_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X_train = X_train.reshape(num_images_train, image_size, image_size)
        
        '''########## READ THE TRAIN LABELS FILE ########'''
        f2 = gzip.open('train-labels-idx1-ubyte.gz','r')
        f2.read(8)
        labels_train=[]
        for i in range(0,60000):   
            buf = f2.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            labels_train.append(np.int(label))
        print("Length of the Y_train list: ",len(labels_train))
        
        '''############# PLOT THE IMAGES ##############'''
        k=98
        import matplotlib.pyplot as plt
        image = np.asarray(X_train[k]).squeeze()
        plt.title("Number: "+str(labels_train[k]))
        plt.imshow(image)
        plt.show()
        
        '''########### READ THE TEST IMAGE FILE #############'''
        import gzip
        f3 = gzip.open('t10k-images-idx3-ubyte.gz','r')

        image_size = 28
        num_images_test = 10000

        f3.read(16)
        buf = f3.read(image_size * image_size * num_images_test)
        X_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X_test = X_test.reshape(num_images_test, image_size, image_size)
        
        '''########## READ THE TEST LABELS FILE ########'''
        f4 = gzip.open('train-labels-idx1-ubyte.gz','r')
        f4.read(8)
        labels_test=[]
        for i in range(0,10000):
            buf = f4.read(1)
            label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            labels_test.append(np.int(label))
        print("Length of the Y_test list: ",len(labels_test))
        
        
        
        '''############## RESHAPING PROCESS OF THE TRAIN DATA ##############'''
        X_train = X_train.reshape((num_images_train, 784))
        X_train = X_train.transpose()
        
        Y_train = np.asarray(labels_train)
        Y_train = Y_train.reshape((num_images_train,1))
        Y_train = Y_train.transpose()
        
        print("\nNumpy array X_train shape: ",X_train.shape)
        print("\nNUmpy array Y_train shape: ",Y_train.shape)
        
        '''#######33###### RESHAPING PROCESS OF THE TEST DATA ################'''
        X_test = X_test.reshape((10000, 784))
        X_test = X_test.transpose()
        
        Y_test = np.asarray(labels_test)
        Y_test = Y_test.reshape((10000,1))
        Y_test = Y_test.transpose()
        
        print("\nNumpy array X_test shape: ",X_test.shape)
        print("\nNumpy array Y_test shape: ",Y_test.shape)
        
        return X_train, Y_train, X_test, Y_test

def softmax(Z):
    '''############ Softmax activation function ###########'''
    '''The function returns two values A and cache. The first to be used for forward propagation and the later to be used during Backprop'''
    
    exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exps/np.sum(exps, axis=0, keepdims=True)

    cache = Z
    
    return A, cache

def relu(Z):
    '''############ Relu Activation Function #############'''
    
    A = np.maximum(0,Z)
    A = np.minimum(10000,A)
    
    assert (A.shape==Z.shape)
    
    cache = Z
    
    return A, cache

def relu_backward(dA, cache):
    ''' ########### Backprop for Single RELU unit ##############'''
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0 ########### when Z<=0, we must put the value of dZ=0 as well #################
    
    assert(dZ.shape == Z.shape)
    
    return dZ

def softmax_backward(dA, cache):
    '''############### Backprop for Single SoftMax Unit #############'''
    
    Z = cache
    
    exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = exps/np.sum(exps, axis=0, keepdims=True)
    
    dZ = dA*A*(1-A)
    
    return dZ

def log_non_zero(AL):
    '''Log value computaion on non zeroes only'''
    p=AL > 0.0
    res=np.zeros_like(AL)
    res[p]=np.log(AL[p])
    
    return res