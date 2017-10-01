import unittest
import numpy as np
from mnist_model import MnistModel

class TestMnistMethods(unittest.TestCase):

    def test_split_data(self):
        X = np.zeros((5,10))
        Y = np.zeros((10,5))
        dev_size = 2
        test_size = 2

        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = MnistModel.split_data(X.T,Y.T, dev_size, test_size)
        print(X_train.shape,X_test.shape,X_dev.shape)
        self.assertEqual(X.shape[1],X_train.shape[1]+X_test.shape[1]+X_dev.shape[1],"Train+Test+Dev should be same size as the dataset")
        #self.assertEqual(Y.shape[1],Y_train.shape[1]+Y_test.shape[1]+Y_dev.shape[1],"Train+Test+Dev should be same size as the dataset")
        self.assertEqual(X_test.shape[1],test_size,"Test set should be same as arguement")
        self.assertEqual(X_dev.shape[1],dev_size,"Dev set should be same as arguement")
        self.assertEqual(X.shape[0],X_train.shape[0], "Training set should have same features as the dataset")
        self.assertEqual(X.shape[0],X_test.shape[0], "Test set should have same features as the dataset")
        self.assertEqual(X.shape[0],X_dev.shape[0], "DEV set should have same features as the dataset")

if __name__ == '__main__':
    unittest.main()
