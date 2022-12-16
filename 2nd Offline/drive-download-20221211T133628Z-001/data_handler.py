import csv
import numpy as np
import pandas as pd

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """

    # todo: implemented
    matrix = []
    vector_of_class = []
    with open("data_banknote_authentication.csv",'r') as file:
        next(file)
        csvreader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC,delimiter=',') 
        for row in csvreader: 
            mat= []
            for entry in range(len(row)-1):
                mat.append(float(row[entry]))
            vector_of_class.append(row[len(row)-1])
            matrix.append(mat)



    return matrix, vector_of_class


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implemented.
    
    

    #print(np.shape(X))
    for i in range(np.shape(X)[0]) : 
        X[i].append(y[i])
    
    if shuffle==True :
        np.random.shuffle(X)

    #print(np.shape(X))
    row_count = int(len(X)*test_size)
    col_count = len(X[0])

    X = np.array(X)
    X_train = X[ : row_count, : col_count-1]
    y_train = X[ : row_count,  col_count-1: col_count]

    """ print(X[:3])
    print(X_train[:3])
    print(y_train[:3]) """

    X_test = X[row_count :, :col_count-1]
    y_test = X[row_count :, col_count-1: col_count]

    
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implemented
    X_sample, y_sample = [], [] 

    while np.shape(X_sample)[0] != np.shape(X)[0]:
        index = np.random.randint(np.shape(X)[0])
        X_sample.append(X[index])
        y_sample.append(y[index]) 

    
    assert len(X_sample) == len(X)
    assert len(y_sample) == len(y)
    return X_sample, y_sample
