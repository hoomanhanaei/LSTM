# sequence_alignment.py

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment, PairwiseAligner


def pairwise_alignment(reference_file, mutated_file, num_mutated_seqs=None):
    """
    Perform pairwise sequence alignment between a reference sequence and mutated sequences.

    Parameters:
    - reference_file (str): Path to the reference sequence file in FASTA format.
    - mutated_file (str): Path to the mutated sequences file in FASTA format.
    - num_mutated_seqs (int): Number of mutated sequences to process (default is 20).

    Returns:
    - alignments (list): List of aligned sequences.
    """
    # Load reference sequence
    reference_seq_record = SeqIO.read(reference_file, "fasta")
    reference_sequence = str(reference_seq_record.seq)

    # Load mutated sequences.
    mutated_seq_records = SeqIO.parse(mutated_file, "fasta")
    if num_mutated_seqs is not None:
        mutated_seq_records = list(SeqIO.parse(mutated_file, "fasta"))[:num_mutated_seqs]

    # Create a PairwiseAligner object
    aligner = PairwiseAligner()

    # Define alignment parameters
    aligner.mode = 'global'  # Perform global alignment
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    alignments = []

    for mutated_seq_record in mutated_seq_records:
        mutated_sequence = str(mutated_seq_record.seq)
        # extract the first alignment which has the highest quality score.
        alignment = aligner.align(reference_sequence, mutated_sequence)[0]

        # ref_aligned_seq = alignment[0]
        mutated_aligned_seq = alignment[1]

        # Check if 'N' characters exist in the mutated sequence
        if 'N' in mutated_sequence:
            # Replace 'N' characters with dashes ('-') in the mutated aligned sequence
            mutated_aligned_seq = mutated_aligned_seq.replace('N', '-')

        alignments.append(mutated_aligned_seq)

    return alignments



def multiple_seq_alignment(padded_sequences):
    """
    Perform multiple sequence alignment on pairwise aligned and padded sequences.

    Parameters:
    - padded_sequences list(str): Path to the pairwise aligned and padded sequences file.txt.

    Returns:
    - multiple sequence alignment (list): List of MSA aligned sequences.
    """

    # Read aligned and padded sequences from a text file.
    aligned_seqs = []

    with open(padded_sequences, "r") as file:
        # Initialize an empty string to store the current sequence
        sequence = ""  
        for line in file:
            line = line.strip()
            if line:
                # Concatenate the line to the current sequence
                sequence += line
            # If the line is empty, it indicates the end of the current sequence
            else:  
                seq_record = SeqRecord(Seq(sequence))
                aligned_seqs.append(seq_record)
                # Reset the sequence string for the next sequence
                sequence = ""  

    # Create a MultipleSeqAlignment object
    alignment = MultipleSeqAlignment(aligned_seqs)

    return alignment



def replace_dashes(alignments, replacement_char): # In case if replacement of dashes with another character is required
    """
    Replace dashes ('-') in sequences with another character.

    Parameters:
    - alignments (list of str): List of input sequences.
    - replacement_char (str): Character to replace dashes with.

    Returns:
    - modified_sequences (list of str): Sequences with dashes replaced by the replacement character.
    """
    modified_sequences = [seq.replace('-', replacement_char) for seq in alignments]

    return modified_sequences




# # test with sample dataset
# if __name__ == "__main__":
#     reference_file = "./data/Wuhan-Hu-1-MN908947.3.fasta"
#     mutated_file = "./data/gisaid_hcov-19_2024_02_22_14.fasta"
#     aligned_sequences = align_sequences(reference_file, mutated_file, num_mutated_seqs=1)
#     for seq in aligned_sequences:
#         print(seq)

