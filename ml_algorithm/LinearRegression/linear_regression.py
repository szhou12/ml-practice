import numpy as np
# from utils.features import prepare_for_training

class LinearRegression:
   
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        Linear regression model
        STEP 1: pre-process data
        STEP 2: count # of features
        STEP 3: initialize param matrix (theta)
        """

        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = data_processed.shape[1]
        self.theta = np.zero((num_features, 1))
    
    def train(self, learning_rae, num_iterations=500):
        pass

    def gradient_descent(self, learning_rate, num_iterations):
        for _ in range(num_iterations):
            self.gradient_step(learning_rate)

    def gradient_step(self, learning_rate):
        """
        One step of gradient descent to update params
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels # residuals
        theta = self.theta 
        theta = theta - learning_rate * (1/num_examples) * (self.data.T @ delta)
        self.theta = theta
    
    @staticmethod
    def hypothesis(data, theta):
        predictions = data @ theta
        return predictions

