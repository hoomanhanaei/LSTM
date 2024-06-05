# # test_model.py

# import os
# import tensorflow as tf


# # Define model checkpoint callback
# save_path = "C:/Users/hanaei/H_H/output"
# checkpoint_path = os.path.join(save_path, "checkpoint")
# os.makedirs(checkpoint_path, exist_ok=True)

# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#                             filepath=os.path.join(checkpoint_path, "model_checkpoint.weights.h5"), 
#                             save_weights_only=True, save_best_only=False
#                             )

# class LSTM(tf.keras.Model):
#     def __init__(self, input_shape, output_shape):
#         super(LSTM, self).__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.lstm_layer = tf.keras.layers.LSTM(units=64)
#         self.dense_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='softmax')
#         self.reshape_layer = tf.keras.layers.Reshape((output_shape[0], output_shape[1]))

#     def call(self, inputs):
#         x = self.lstm_layer(inputs)
#         x = self.dense_layer(x)
#         x = self.reshape_layer(x)
#         return x
