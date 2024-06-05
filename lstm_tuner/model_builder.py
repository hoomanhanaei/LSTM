# lstm_tuner/model_builder.py



from model.myLSTM import LSTM
import tensorflow as tf



def build_model(hp, input_shape=None, output_shape=None):
    """
    Builds and compiles an LSTM model based on the different hyperparameters.

    Args:
        hp (keras_tuner.HyperParameters): A HyperParameters object containing\
                                        the hyperparameters for model configuration.
        input_shape (tuple): The shape of the input data. Defaults to None.
        output_shape (tuple): The shape of the output data. Defaults to None.

    Returns:
        tensorflow.keras.Model: A compiled LSTM model.
    """
    model = LSTM(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        input_shape=input_shape,
        output_shape=output_shape,
        dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.2, step=0.1)
        )
    
    optimizer = hp.Choice('optimizer', ['adafactor','sgd', 'adam',
                                        'nadam','adagrad','adadelta',
                                        'adamax','lion']
                          )
    # metrics = hp.Choice('metrics', [keras.metrics.Accuracy,
    #                                 keras.metrics.categorical_accuracy]
    #                     )

    # metrics_name = hp.Choice('metrics', ['Accuracy', 'categorical_accuracy'])
    # metrics = [getattr(tf.keras.metrics, metrics_name)()]

    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics= ["accuracy"]
                )
    
    return model





# def build_model(hp, input_shape=None, output_shape=None):
#     """
#     Builds and compiles an LSTM model based on the different hyperparameters.

#     Args:
#         hp (keras_tuner.HyperParameters): A HyperParameters object containing\
#                                         the hyperparameters for model configuration.
#         input_shape (tuple): The shape of the input data. Defaults to None.
#         output_shape (tuple): The shape of the output data. Defaults to None.

#     Returns:
#         tensorflow.keras.Model: A compiled LSTM model.
#     """
#     model = LSTM(
#         units=hp.Int('units', min_value=32, max_value=256, step=32),
#         input_shape=input_shape,
#         output_shape=output_shape,
#         dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1),
#         learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]),
#         num_layers=hp.Int('num_layers', min_value=1, max_value=3),
#         activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']),
#         kernel_regularizer=hp.Choice('kernel_regularizer', values=['l1', 'l2', None]),
#         recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.0, max_value=0.5, step=0.1),
#         bidirectional=hp.Boolean('bidirectional')
#     )
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model.learning_rate),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
    
#     return model
