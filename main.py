
import logging
import argparse
import os
import time
import numpy as np

from modules import data_loader as dl
from modules import sequence_padder as sp
from modules import encoder_decoder as ed
from modules import window_generator as wg

from lstm_tuner.tuner import run_tuner
from lstm_tuner.model_builder import build_model
# from model import myLSTM



def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform LSTM.")
    parser.add_argument("--ref_path", help="Path to the reference sequence file in FASTA format")
    parser.add_argument("--mut_path", help="Path to the mutated sequences file in FASTA format")
    parser.add_argument("--num_mut_seqs", type=int, help="Number of mutated sequences to process. If not provided, process all sequences.")
    parser.add_argument("--out_dir", default= './output', help="Output directory to save aligned sequences.")

    args = parser.parse_args()
    # logging configuration
    logging.basicConfig(level = logging.INFO, format='%(message)s')

    logging.info("\nLoading sequences... (Start Time: %s)", time.strftime('%Y-%m-%d %H:%M:%S'))
    sequences = dl.load_sequences_from_fasta(args.mut_path, args.ref_path,  args.num_mut_seqs)
    num_seqs = len(sequences)
    logging.info("%s\nTotal Sequences: \n%d \n%s", '-' * 40, num_seqs, '-' * 40)

    # Perform padding on aligned sequences
    logging.info("\nSequence Padding... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    padded_sequences, max_length, num_padded, num_shortened = sp.pad_sequences(args.ref_path, sequences)
    
    logging.info("Reference Sequence Length: %d", max_length)
    logging.info("Number of Mutated Sequences Padded: %d", num_padded)
    logging.info("Number of Mutated Sequences Shortened: %d", num_shortened)    
    # logging.info("Padded Sequences saved to: \n%s \n%s", output_file_padded, '-' * 40)

    logging.info("\nValid nucleotides filtering... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    valid_char = {'A', 'C', 'T', 'G', '-'}
    train_sequences, validation_sequences = dl.filter_valid_nucleotides(padded_sequences, valid_char)
    logging.info("\n%s", '-' * 40)

    logging.info("\nNucleotides encoding... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    encoder = ed.EncoderDecoder()
    train_encoded = encoder.one_hot_encode(train_sequences)
    validation_encoded = encoder.one_hot_encode(validation_sequences)
    logging.info("Shape of encoded sequences: of\n%s \n%s", np.asarray(train_encoded).shape,'-' * 40)


    logging.info("\nReshaping data... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    # Reshape to 2D
    reshaped_data = np.reshape(train_encoded, (-1, np.asarray(train_encoded).shape[2]))
    logging.info("Encoded sequences reshaped to:\n%s \n%s", reshaped_data.shape,'-' * 40)

    logging.info("\nCreating training windows... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    input_width = np.asarray(train_encoded).shape[1]*3  # 96 nucleotides as input
    label_width = np.asarray(train_encoded).shape[1] # Predict 32 nucleotides or a complete sequence
    shift = np.asarray(train_encoded).shape[1]  # Shift of 32 nucleotides

    w2 = wg.WindowGenerator(input_width=input_width,
                            label_width=label_width,
                            shift=shift
                            )
    logging.info("Windows created with the following size:\n%s \n%s",w2,'-' * 40)

    logging.info("\nCreating batches of datasets containing multiple windows... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    trn = w2.make_dataset(reshaped_data)
    vali = w2.make_dataset(reshaped_data)
    logging.info("Generated datasets: \nInput dataset:\n%s \nTarget dataset:\n%s\
                  \nTotal number of datasets:\n%s \n%s",trn.element_spec[0], trn.element_spec[1],len(trn),'-' * 40)
    
    logging.info("\nHyperparameter tuning... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    input_shape = (np.asarray(train_encoded).shape[1]*3, np.asarray(train_encoded).shape[2])
    output_shape = (np.asarray(train_encoded).shape[1], np.asarray(train_encoded).shape[2])

    tuner = run_tuner(trn, validation_data=vali, input_shape=input_shape, output_shape=output_shape)
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info("\nModel training... (Start Time: %s)\n%s",
                    time.strftime('%Y-%m-%d %H:%M:%S'),'-' * 40)
    best_model = build_model(best_hps, input_shape=input_shape, output_shape=output_shape)
    history = best_model.fit(trn, epochs=10)  # Adjust epochs and other parameters as needed
    # model = myLSTM.LSTM(input_shape,output_shape)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(x, epochs=10, callbacks=[myLSTM.checkpoint_callback])



# 
if __name__ == "__main__":
    main()


