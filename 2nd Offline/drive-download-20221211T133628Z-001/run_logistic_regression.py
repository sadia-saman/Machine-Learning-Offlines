"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import precision_score, recall_score, f1_score,accuracy
import numpy as np

if __name__ == '__main__': 
    # data load
    X, y = load_dataset() 
    
    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y,0.7,True)  
    

    # training
    params = dict({'learning_rate' : 0.1, 'iteration' : 100})
    classifier = LogisticRegression(params)
    classifier.fit(X_train, y_train)

    # testing
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    y_pred = classifier.predict(X_test) 



    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
