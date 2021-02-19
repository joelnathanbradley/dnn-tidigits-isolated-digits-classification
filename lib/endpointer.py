from sklearn.mixture import GaussianMixture
import numpy as np


class Endpointer:
    """
    Endpointer
    Unsupervised voice activity detector
    
    Uses a GMM to learn a two class distribution of RMS energy vectors
    
    """

        
    def __init__(self, train):
        """
        Endpointer(iterable) - Create an endpointer that is trained from
        an instance of a class that iterates to produce features

        :param train:  Numpy N x D matrix of N examples of D dimensional features

        """
        # create and fit gaussian 2 mixture model
        self.model = GaussianMixture(n_components=2, covariance_type='diag').fit(train)



    def predict(self, features):
        """
        predict
        :param features: Numpy N x D matrix of N examples of D dimensional features
        :return: binary vector of length N, True for frames classified as speech
        """

        # make predictions by categorizing frames as speech or silence
        predictions = self.model.predict(features)
        # If silence gaussian mixture fit with 1 as the category, swap the classifications so that speech has
        # category of 1, and silence has category of 0.
        means = self.model.means_
        if np.mean(means[0]) > np.mean(means[1]):
            predictions = 1 - predictions
        return predictions

        
            
    
