from data_handler import bagging_sampler
from linear_model import LogisticRegression
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # ...................implemented........................................
        self.base_estimator = base_estimator 
        self.n_estimator = n_estimator
        self.Thetas = []
        self.losses = []
        

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # ...................................implemented..................................
        
        for i in range(self.n_estimator):
            temp_X, temp_y = bagging_sampler(X,y)
            print("n_estimator ",i)
            self.base_estimator.fit(temp_X,temp_y)
            self.Thetas.append(self.base_estimator.theta)

            
            temp_X = np.concatenate((np.ones((np.shape(temp_X)[0], 1)), temp_X), axis=1)
            h = self.base_estimator.predict(temp_X)
            self.losses.append(self.base_estimator.loss(h,temp_y))
            
            


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # ................implemented...................
        loss = loss = np.min(self.losses)

        for i in range(len(self.losses)):
            if self.losses[i]==loss :
                loss = self.losses[i]
                self.base_estimator.theta = self.Thetas[i]
                break
        
        prediction = self.base_estimator.sigmoid(X)

        for i in range(len(prediction)):
            if(prediction[i]>=0.5) :
                prediction[i] = 1
            else:
                prediction[i] = 0
        
        return prediction
 