# myLSTM.py

import os
import tensorflow as tf


# Define model checkpoint callback
save_path = "C:/HOOMAN/phase_02/LSTM/output/model"
checkpoint_path = os.path.join(save_path, "checkpoint")
os.makedirs(checkpoint_path, exist_ok=True)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=os.path.join(checkpoint_path, "model_checkpoint.weights.h5"), 
                            save_weights_only=True, save_best_only=False
                            )

# LSTM class
class LSTM(tf.keras.Model):
    """
    Long Short-Term Memory (LSTM) model.
    This class represents an LSTM model built using TensorFlow's Keras API.
    
    Attributes:
        input_shape (tuple): The shape of the input data (batch_size, sequence_length, input_dimension).
        output_shape (tuple): The shape of the output data (batch_size, sequence_length, output_dimension).
    """
    def __init__(self, units, input_shape, output_shape,
                dropout_rate,
                ):
        """
        Initializes the LSTM model and take two arguments.
        """
        super(LSTM, self).__init__()
        """
        calls the constructor of the superclass (tf.keras.Model).
        performs any necessary initialization, to ensure that the LSTM class inherits all properties\
        and methods from tf.keras.Model.
        """
        self.units = units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lstm_layer = tf.keras.layers.LSTM(units)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)  # Add dropout layer
        self.dense_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='softmax')
        self.reshape_layer = tf.keras.layers.Reshape((output_shape[0], output_shape[1]))

    def call(self, inputs):
        """
        Performs forward pass through the LSTM model.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, sequence_length, input_dimension).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, sequence_length, output_dimension).
        """
        x = self.lstm_layer(inputs)
        x = self.dropout_layer(x)  # Apply dropout after LSTM layer
        x = self.dense_layer(x)
        x = self.reshape_layer(x)

        return x