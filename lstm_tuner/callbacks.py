# lstm_tuner/callbacks.py

import os
from tensorflow.keras.callbacks import Callback



class MetricsLogger(Callback):
    """
    A Keras callback for logging metrics at the end of each epoch for each trial.

    Attributes:
        log_dir (str): The directory path where log files will be stored.
    """
    def __init__(self, log_dir):
        super(MetricsLogger, self).__init__()
        self.log_dir = log_dir
        self.trial_epoch_metrics = []
        self.trial_id = 0


    # def on_train_begin(self, logs=None):
    #     """
    #     Callback function called by Keras at the beginning of training.

    #     Args:
    #         logs (dict): A dictionary containing the training metrics for the entire training process.
    #     """
    #     if logs is None:
    #         logs = {}

    #     # # Get the trial number from the tuner
    #     # trial_number = self.model.stop_training
    #     # self.trial_id = trial_number
    #     # Reset trial-specific metrics for the next trial
    #     self.trial_id += 1



    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called by Keras at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing the training metrics for the epoch.
        """
        if logs is None:
            logs = {}

        # Store metrics for the current epoch
        self.trial_epoch_metrics.append(logs)



    def on_train_end(self, logs=None):
        """
        Callback function called by Keras at the end of training.

        Args:
            logs (dict): A dictionary containing the training metrics for the entire training process.
        """
        if logs is None:
            logs = {}

        # Save metrics for each trial
        self._save_trial_metrics()
        self.trial_id += 1

        

    def _save_trial_metrics(self):
        """
        Save the metrics for each trial to a separate file.
        """
        # trial_id = len(self.trial_epoch_metrics)
        trial_file = os.path.join(self.log_dir, f"trial_{self.trial_id}_metrics.txt")

        # Create the directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        with open(trial_file, 'w') as f:
            for epoch, metrics in enumerate(self.trial_epoch_metrics, start=1):
                f.write(f"Epoch {epoch} metrics:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")





