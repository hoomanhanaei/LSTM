import os
import time
import shutil

def observe_and_move_logs(log_dir, destination_dir):
    """
    Observes the log directory for any new files and moves them to a new directory with appropriate names.

    Args:
        log_dir (str): Directory path where log files are generated.
        destination_dir (str): Directory path where new trial log files will be stored.
    """
    while True:
        # List all files in the log directory
        files = os.listdir(log_dir)
        
        # Filter out only the files with .txt extension
        txt_files = [f for f in files if f.endswith('.txt')]
        
        # Sort the files by creation time
        txt_files.sort(key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
        
        # Count the number of files already in the destination directory
        num_existing_files = len(os.listdir(destination_dir))
        
        # Move each file to the destination directory with appropriate name
        for i, txt_file in enumerate(txt_files, start=num_existing_files + 1):
            new_name = f"trial_{i}_log.txt"
            src_path = os.path.join(log_dir, txt_file)
            dst_path = os.path.join(destination_dir, new_name)
            shutil.move(src_path, dst_path)
            print(f"Moved {txt_file} to {new_name}")
        
        # Sleep for a while before checking again
        time.sleep(1)

# Example usage
if __name__ == "__main__":
    log_dir = "logs"  # Your original log directory
    destination_dir = "trial_logs"  # Directory where trial log files will be moved
    os.makedirs(destination_dir, exist_ok=True)  # Create destination directory if it doesn't exist
    observe_and_move_logs(log_dir, destination_dir)
