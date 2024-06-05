# sequence_encoder.py

import numpy as np
from Bio import SeqIO
import tensorflow as tf


def load_sequences_from_fasta(mutated_file, reference_file=None):
    """
    Retrieves and Load sequences from files.

    Parameters:
        mutated_file (str): Path to the FASTA file containing mutated sequences.
        reference_file (str, optional): Path to the FASTA file containing reference sequences.
                                        Defaults to None.

    Returns:
        list: A list of sequences extracted from the provided FASTA files.
    """

    sequences = []
    
    # Load sequences from the reference FASTA file
    with open(mutated_file, 'r') as file:
        records = list(SeqIO.parse(file, "fasta"))[:1]
        for record in records:
            sequences.append(str(record.seq))
    
    # If a reference file is provided, append its sequence to the list
    if reference_file:
        with open(reference_file, 'r') as ref_file:
            for record in SeqIO.parse(ref_file, "fasta"):
                sequences.append(str(record.seq))

    return sequences



def filter_valid_nucleotides(sequences):
    """
    Filters out invalid nucleotides from a list of sequences.

    This function replaces any nucleotide that is not in the below list\
    ['A', 'C', 'T', 'G', or '-' with '-'] --> *this list will be updated*.

    Parameters:
        sequences (list): A list of genomic sequences.

    Returns:
        list: A list of preprocessed sequences where invalid nucleotides are replaced with '-'.
    """
    preprocessed_sequences = []
    for sequence in sequences:
        preprocessed_sequence = ''.join([
                nucleotide if nucleotide in {'A', 'C', 'T', 'G', '-'} else '-' for nucleotide in sequence
                ])
        preprocessed_sequences.append(preprocessed_sequence)

    return preprocessed_sequences



# Define the nucleotide mapping
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}

def one_hot_encode_sequence(sequence):
    """
    One-hot encodes a nucleotide sequence.
    
    Args:
    - sequence (str): The nucleotide sequence to encode.
    
    Returns:
    - numpy array: The one-hot encoded representation of the sequence.
    """
    encoded_sequence = np.zeros((len(sequence), 5), dtype=np.uint8)  # Initialize an array of zeros
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotide_map:
            encoded_sequence[i, nucleotide_map[nucleotide]] = 1  # Set the corresponding position to 1
    return encoded_sequence




