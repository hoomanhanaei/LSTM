from Bio import SeqIO



def pad_sequences(reference_file, sequences, padding_value='-'):
    """
    Pad sequences to ensure they have the same length as the reference sequence.

    Parameters:
    - reference_sequence (str): Reference sequence.
    - sequences (list of str): List of aligned sequences.
    - padding_value (str): Value used for padding.

    Returns:
    - padded_sequences (list of str): Padded sequences.
    """

    # Load reference sequence
    reference_seq_record = SeqIO.read(reference_file, "fasta")
    reference_sequence = str(reference_seq_record.seq)

    max_length = len(reference_sequence)
    padded_sequences = []
    num_padded = 0
    num_shortened = 0
    
    for seq in sequences:
        original_length = len(seq)
        if original_length < max_length:
            padded_seq = seq + (max_length - original_length) * padding_value
            num_padded += 1

        else:
            padded_seq = seq[:max_length]  # Trim sequences longer than max_length
            num_shortened += 1
        padded_sequences.append(padded_seq)
    
    return padded_sequences, max_length, num_padded, num_shortened