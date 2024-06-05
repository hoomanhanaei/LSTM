# lstm_tuner/tuner.py
import os
import shutil

from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch, GridSearch

from .callbacks import MetricsLogger
from .model_builder import build_model



def run_tuner(train_data, validation_data, input_shape, output_shape, log_dir='logs', max_trials=20):
    """
    Runs hyperparameter tuning for our LSTM model using Keras Tuner.

    Args:
        train_data (numpy.ndarray or tensorflow.data.Dataset): Training data.
        validation_data (numpy.ndarray or tensorflow.data.Dataset): Validation data.
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output data.
        log_dir (str, optional): Directory to store logs. Defaults to 'logs'.
        max_trials (int, optional): Maximum number of hyperparameter configurations to test. Defaults to 5.

    Returns:
        keras_tuner.tuners.RandomSearch: Tuner object containing the results of the hyperparameter search.
    """
    
    # Delete the directory containing tuner checkpoints
    # shutil.rmtree('my_dir/lstm_tuning', ignore_errors=True)

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    

    # Define the search space
    tuner = GridSearch(
        lambda hp: build_model(hp, input_shape=(input_shape,), output_shape=output_shape),
        objective='val_accuracy',
        max_trials=max_trials,
        directory='my_dir',
        project_name='lstm_tuning')

    # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Start the search for the best hyperparameters with custom callback
    
    tuner.search(train_data, validation_data=validation_data,
                epochs=10, callbacks=[MetricsLogger(log_dir)])

    return tuner