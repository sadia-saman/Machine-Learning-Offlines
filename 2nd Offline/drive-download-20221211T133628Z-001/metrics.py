"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #formula = (tp + tn)/(tp + tn + fp + tn)
    #print(sum(condition(x) for x in lst))
    
    """ tp = np.sum(np.logical_and(y_true==1 , y_pred==1))
    tn = np.sum(np.logical_and(y_true==0 , y_pred ==0)) """

    tp = 0
    tn = 0
    for i in range(len(y_true)): 
        if y_true[i][0]==1 and y_pred[i]==1:
            tp = tp+1
        elif y_true[i][0]==0 and y_pred[i]==0:
            tn = tn+1

    return float((tp + tn)/np.shape(y_true)[0])
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    # formula = tp / (tp + fp)
    """  tp = np.sum(np.logical_and(y_true==1 , y_pred==1))
    fp = np.sum(np.logical_and(y_true==0 , y_pred==1)) """
    tp = 0
    fp = 0
    for i in range(len(y_true)): 
        if y_true[i][0]==1 and y_pred[i]==1:
            tp = tp+1
        elif y_true[i][0]==0 and y_pred[i]==1:
            fp = fp+1
    
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    # formula = tp / (tp + fn)

    """ tp = np.sum(np.logical_and(y_true==1 , y_pred==1))
    fn = np.sum(np.logical_and(y_true==1 , y_pred==0)) """
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i][0]==1 and y_pred[i]==1:
            tp = tp+1
        elif y_true[i][0]==1 and y_pred[i]==0:
            fn = fn+1

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #formula : F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    recall = recall_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred)
    return 2*(recall * precision) / (recall + precision)
