# data_loader.py

from Bio import SeqIO
from sklearn.model_selection import train_test_split



def load_sequences_from_fasta(mutated_file, reference_file = None, num_mutated_seqs=None):
    """
    Load sequences from FASTA files.

    Parameters:
    - mutated_file (str): Path to the FASTA file containing mutated sequences.
    - reference_file (str, optional): Path to the reference FASTA file. Defaults to None.
    - num_mutated_seqs (int, optional): Number of mutated sequences to load. Defaults to None.

    Returns:
    - sequences (list): List of sequences from mutated_file and reference_file if provided.
    """

    sequences = []
    
    # If a reference file is provided, append its sequence to the list
    if reference_file:
        with open(reference_file, 'r') as ref_file:
            for record in SeqIO.parse(ref_file, "fasta"):
                sequences.append(str(record.seq))
    
    # Load sequences from the mutated FASTA file
    with open(mutated_file, 'r') as file:
        if num_mutated_seqs is not None:
            records = list(SeqIO.parse(file, "fasta"))[:num_mutated_seqs]
        else:
            records = list(SeqIO.parse(file, "fasta"))
        for record in records:
            sequences.append(str(record.seq))
    
    return sequences



def filter_valid_nucleotides(sequences, valid_nucleotides, replacement='-', test_size=0.2, random_state=None):
    """
    Filters out invalid nucleotides from a list of sequences.

    This function replaces any nucleotide that is not in the provided dictionary of valid nucleotides
    with the specified replacement character.

    Parameters:
        sequences (list): A list of genomic sequences.
        valid_nucleotides (set): Set of valid nucleotides. (e.g., {'A', 'C', 'T', 'G', '-'}).
        replacement (char, optional): Character to replace invalid nucleotides. Defaults to '-'.

    Returns:
        list: A list of filtered sequences where ambiguous nucleotides are replaced with the specified character.
    """
    # Print a list of all unique characters of sequences
    all_characters = set(''.join(sequences))
    print("List of all unique characters of sequences:", all_characters)

    filtered_sequences = []
    for sequence in sequences:
        filtered_sequence = ''.join([
                nucleotide if nucleotide in valid_nucleotides else replacement for nucleotide in sequence.upper()
                ])
        filtered_sequences.append(filtered_sequence)

    # Print a list of unique nucleotides after filtering
    filtered_characters = set(''.join(filtered_sequences))
    print("List of unique nucleotides after filtering:", filtered_characters)

    # Split filtered sequences into train and validation sets
    train_sequences, validation_sequences = train_test_split(filtered_sequences, test_size=test_size, random_state=random_state)

    return train_sequences, validation_sequences