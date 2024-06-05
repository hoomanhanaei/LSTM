# window_generator.py
#####################################################
# NOTICE:
# the following script is a modified version of a the the code accessible in this address:
# https://www.tensorflow.org/tutorials/structured_data/time_series
#####################################################

import numpy as np
import tensorflow as tf



class WindowGenerator():
    '''
    ** WindowGenerator class is modified to accept univariate time series data and create input and label
    windows for model training and validation.
    Since encoding the genomic data using tensorflow, shapes the data into a tensor itself,
    and the offset for prediction validation will be an entire genome, the width of both label and input\
    should be the same.
    instead of using window generator class another module was used to create\
    a batched dataset of the tensor data.
    Important if the tf.one_hot was used to encode the data, data should be converted\
    to np.array and reshaped to two dimensional before using this class.
    
    Args:
        input_width (int): Width of the input window.
        label_width (int): Width of the label window.
        shift (int): Number of time steps between successive windows.
        input_sequences (int): Number of sequences in the input.
    
    Attributes:
        total_window_size (int): Total size of the window including input width and shift.
        input_slice (slice): Slice object for input window.
        input_indices (numpy.ndarray): Indices for input window.
        label_start (int): Start index of the label window.
        labels_slice (slice): Slice object for label window.
        label_indices (numpy.ndarray): Indices for label window.

    '''


    def __init__(self, input_width, label_width, shift):

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = tf.range(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = tf.range(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        '''
        Returns a string representation of the WindowGenerator object.
        '''
        
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {len(self.input_indices)}',
            f'Label indices: {len(self.label_indices)}'])


    def split_window(self, features):
        '''
        Splits the time steps containing the features into input and label windows.
        
        Args:
            features: dataset containing the the time steps or time series.
        
        Returns:
            tuple (tensor): tensor contains 3 axes batch, time step, features inside time step.
        '''
        
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def make_dataset(self, data):
        '''
        Creates a TensorFlow dataset from the given data, suitable for training or evaluation.

        Args:
            data (array-like): The input data, typically a time series or sequence data.

        Returns:
            tf.data.Dataset: A TensorFlow dataset containing input-output pairs, where each
            element is a tuple of tensors representing inputs and labels, prepared according
            to the window configuration.

        Notes:
        The function first converts the input data into a numpy array of dtype np.float32.
        Then, it constructs a TensorFlow dataset using the provided data, dividing it into
        windows of size `total_window_size` with a stride of `shift`(to account for a whole sequence length).
        Each window is divided into input and label portions based on the window configuration, and the resulting
        dataset is not shuffled (since positions are important) and is batched for training or evaluation.
        
        '''
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.shift,
            shuffle=False,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds




