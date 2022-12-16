import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.n_iteration = params['iteration']
        self.alpha = params['learning_rate']
        self.theta = []
        # todo: implement


    #..........................some additional functions............


    def sigmoid(self, X):
        z = np.dot(X, self.theta) 
        return 1 / (1 + np.exp(-z))


    # this loss function here is to calculate how bad is our prediction

    def loss(self, h, y):
        h_ = [1-h_i for h_i in h]
        y_ = [1-y_i for y_i in y]
        
        loss = []
        for i in range(len(y)):
            if h[i]==0:
                loss.append(-(y_[i]*np.log(h_[i])))
            elif h_[i]==0 :
                loss.append(((-y[i])*np.log(h[i])))
            else:
                loss.append(((-y[i])*np.log(h[i]))-(y_[i]*np.log(h_[i])))

        return np.mean(loss)

    
    # Since we are calculating loss , we want to update however the heck we were learning 
    # about the deciding factors
    # So,This creepy function is here to figure out which direction we are going
    # Means, whether to add stuff to stuff, or subtract stuff from stuff
    def gradient_descent(self, X, h, y):
        h_y = [a_i - b_i for a_i, b_i in zip(h, y)] 
        return (np.dot(X.T, (h_y))*2)/ y.shape[0]


    #Here comes the updating 
    def update_theta(self,gradient):
        res = [t_i - self.alpha*g_i for t_i, g_i in zip(self.theta, gradient)]
        self.theta =  res


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
         
        X = np.array(X)
        y = np.array(y)
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # todo: implement
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iteration):
            h = self.predict(X)
            gradient = self.gradient_descent(X, h, y)
            self.update_theta(gradient) 



    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement 
        res = self.sigmoid(X)
        y_pred = []
        
        for x in res:
            if(x>=0.5):
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred

    