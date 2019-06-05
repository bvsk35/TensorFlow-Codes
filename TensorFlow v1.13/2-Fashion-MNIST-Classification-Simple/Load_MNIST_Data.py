# This file prepares the training, validation and testing MNIST dataset
# Import required functions
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class MNIST(object):
    def __init__(self, scaling='MMS'):
        # No. of pixels in each dimension
        self.img_size = 28
        
        # Size of the total image and
        # the images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_size * self.img_size
        
        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)
        
        # Number of colour channels for the images: 1 channel for gray-scale.
        self.num_channels = 1

        # Tuple with height, width and depth used to reshape arrays.
        # This is used for reshaping in Keras.
        self.img_shape_full = (self.img_size, self.img_size, self.num_channels)
        
        # Number of classes, one class for each of 10 digits.
        self.num_classes = 10
        
        # Number of images in each sub-set.
        self.num_train = 55000
        self.num_val = 5000
        self.num_test = 10000
        
        # How the data should be scaled?
        # SS: Standard Scaler removes mean i.e., makes mean zero and features of the data to hvae unit variance
        # MMS: MinMaxScaler scales the features of the data to lie between [0, 1]
        # For more on this refer this: 
        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        self.scaling = scaling
        
        
    # Function for loading images and labels 
    def load_images(self, filename):
        with gzip.open(filename) as f:
            a = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
            return a

    def load_labels(self, filename):
        with gzip.open(filename) as f:
            a = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            return a
    
    # Function for preparing data
    def prep_data(self, train_img_filename, train_label_filename, test_img_filename, test_label_filename):
        # Load the data
        train_img = self.load_images(train_img_filename)
        train_label = self.load_labels(train_label_filename)
        test_img = self.load_images(test_img_filename)
        test_label = self.load_labels(test_label_filename)
        
        # Convert the unit8 data of images to float64 and 
        # convert the byte data of labels to int
        train_img = train_img.astype(dtype=float)
        test_img = test_img.astype(dtype=float)
        train_label = train_label.astype(dtype=int)
        test_label = test_label.astype(dtype=int)
        
        # Scale the data: so that features have zero mean & unit variance or make features lie between [0, 1]
        if self.scaling == 'SS':
            scaler = StandardScaler().fit(train_img)
            train_img_scaled = scaler.transform(train_img)
            test_img_scaled = scaler.transform(test_img)
        elif self.scaling == 'MMS':
            scaler = MinMaxScaler().fit(train_img)
            train_img_scaled = scaler.transform(train_img)
            test_img_scaled = scaler.transform(test_img)
        
        # Split data into Training, Validation, Testing
        train_img_set = train_img_scaled[:self.num_train]
        train_label_set = train_label[:self.num_train]
        val_img_set = train_img_scaled[self.num_train:]
        val_label_set = train_label[self.num_train:]
        test_img_set = test_img_scaled
        test_label_set = test_label
        
        return train_img_set, train_label_set, val_img_set, val_label_set, test_img_set, test_label_set
    
    # Function for converting the labels to one hot encoded vectors
    def one_hot_encoded(self, label_train, label_val, label_test):
        hot_label_train = []
        hot_label_val = []
        hot_label_test = []
        # One hot encode the training labels 
        for i in range(self.num_train):
            hot_label_train.append(np.eye(self.num_classes, dtype=float)[label_train[i]])
        hot_label_train = np.array(hot_label_train)
        
        # One hot encode the validation labels 
        for i in range(self.num_val):
            hot_label_val.append(np.eye(self.num_classes, dtype=float)[label_val[i]])
        hot_label_val = np.array(hot_label_val)
        
        # One hot encode the testing labels 
        for i in range(self.num_test):
            hot_label_test.append(np.eye(self.num_classes, dtype=float)[label_test[i]])
        hot_label_test = np.array(hot_label_test)
        
        return hot_label_train, hot_label_val, hot_label_test
        
        
    # Function for creating random batches for training
    def random_batch(self, train_img_set, train_label_set, train_hot_set, batch_size=32):
        # Select random indices from training set equivalent to the batch size
        idx = np.random.randint(low=0, high=self.num_train, size=batch_size)
        
        # Prepare the training batched accordingly 
        train_img_batch = train_img_set[idx]
        train_label_batch = train_label_set[idx]
        train_hot_batch = train_hot_set[idx]
        
        return train_img_batch, train_label_batch, train_hot_batch