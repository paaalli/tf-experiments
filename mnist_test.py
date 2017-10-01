import unittest
import numpy as np
from mnist_model import MnistModel
from tensorflow.python.framework import ops

class TestMnistMethods(unittest.TestCase):

    def test_split_data(self):

        # Mock data
        mm = MnistModel()
        X = np.zeros((10,50)) # 10 features, 50 samples
        Y = np.zeros((5,50)) # 5 multi-labels
        dev_size = 5
        test_size = 10

        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = mm.split_data(X,Y, dev_size, test_size)

        self.assertEqual(X.shape[1],X_train.shape[1]+X_test.shape[1]+X_dev.shape[1],"Train+Test+Dev samples should be same size as the dataset")
        self.assertEqual(Y.shape[1],Y_train.shape[1]+Y_test.shape[1]+Y_dev.shape[1],"Train+Test+Dev samples should be same size as the dataset")
        self.assertEqual(X_test.shape[1],test_size,"Test set should be same as arguement")
        self.assertEqual(Y_test.shape[1],test_size,"Test set should be same as arguement")
        self.assertEqual(X_dev.shape[1],dev_size,"Dev set should be same as arguement")
        self.assertEqual(Y_dev.shape[1],dev_size,"Dev set should be same as arguement")
        self.assertEqual(X.shape[0],X_train.shape[0], "Training set should have same features as the dataset")
        self.assertEqual(Y.shape[0],Y_train.shape[0], "Training set should have same features as the dataset")
        self.assertEqual(X.shape[0],X_test.shape[0], "Test set should have same features as the dataset")
        self.assertEqual(Y.shape[0],Y_test.shape[0], "Test set should have same features as the dataset")
        self.assertEqual(X.shape[0],X_dev.shape[0], "DEV set should have same features as the dataset")
        self.assertEqual(Y.shape[0],Y_dev.shape[0], "DEV set should have same features as the dataset")
        ops.reset_default_graph()

    def test_initialize_parameters(self):
        mm = MnistModel()
        layers_dims = [10,20,30]
        params = mm.initialize_parameters(layers_dims)
        self.assertEqual(params['W1'].shape,(layers_dims[1],layers_dims[0]), "Weight matrix shape in the first hidden layer should match the input")
        self.assertEqual(params['b1'].shape,(layers_dims[1],1), "Bias vector shape in the first hidden layer should match the inputs")
        self.assertEqual(params['W2'].shape,(layers_dims[2],layers_dims[1]), "Weight matrix shape in the first hidden layer should match the input")
        self.assertEqual(params['b2'].shape,(layers_dims[2],1), "Bias vector shape in the first hidden layer should match the inputs")
        self.assertEqual(len(params.keys()),(len(layers_dims)-1)*2, "Parameters should be two times the number of hidden layers")
        ops.reset_default_graph()

    def test_random_mini_batches(self):
        mm = MnistModel()
        nr_samples = 53
        nr_features = 12
        nr_labels = 5
        X = np.zeros((nr_features,nr_samples)) # 12 features, 53 samples
        Y = np.zeros((nr_labels,nr_samples)) # 5 multi-labels
        minibatch_size = 10
        mini_batches = mm.random_mini_batches(X, Y, minibatch_size)
        X1 = mini_batches[0][0]
        Y1 = mini_batches[0][1]
        X_last = mini_batches[-1][0]
        Y_last = mini_batches[-1][1]
        self.assertEqual(mini_batches.shape[0],6,"Number of batches should be 6")
        self.assertEqual(mini_batches.shape[1],2,"Each minibatch should contain X and Y data")
        self.assertEqual(X1.shape[0],nr_features,"Number of features should be same in the batch as the input data")
        self.assertEqual(X1.shape[1],minibatch_size,"Number of samples in the first batch should be equal to batch size")
        self.assertEqual(Y1.shape[0],nr_labels,"Number of labels should be same in the batch as the input data")
        self.assertEqual(Y1.shape[1],minibatch_size,"Number of samples in the first batch should be equal to batch size")
        self.assertEqual(X_last.shape[0],nr_features,"Number of features should be same in the last batch as the input data")
        self.assertEqual(X_last.shape[1],nr_samples % minibatch_size,"Number of samples in the last batch should be equal to 3")
        self.assertEqual(Y_last.shape[0],nr_labels,"Number of features should be same in the last batch as the input data")
        self.assertEqual(Y_last.shape[1],nr_samples % minibatch_size,"Number of samples in the last batch should be equal to 3")
        ops.reset_default_graph()

if __name__ == '__main__':
    unittest.main()
