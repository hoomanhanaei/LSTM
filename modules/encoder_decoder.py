import numpy as np

class EncoderDecoder:

    def __init__(self):
        '''
        Initializes an instance of EncoderDecoder.

        Attributes:
        - unique_nucleotides (list or None): A list containing unique nucleotides found in the sequences.
        - nucleotide_map (dict or None): A dictionary mapping each unique nucleotide to its index in the one-hot encoding.
        - sequence_length (int or None): The length of sequences being encoded.
        '''
        self.unique_nucleotides = None
        self.nucleotide_map = None
        self.sequence_length = None


    def one_hot_encode(self, sequences):
        '''
        One-hot encodes a list of nucleotide sequences.

        Parameters:
        - sequences (list): List of nucleotide sequences to encode.

        Returns:
        - encoded_sequences (list): List of one-hot encoded sequences.
        '''
        
        # Extract unique nucleotides
        all_nucleotides = ''.join(sequences)
        self.unique_nucleotides = sorted(set(all_nucleotides))
        print("Unique nucleotides:", self.unique_nucleotides)

        # Generate nucleotide mapping dictionary
        self.nucleotide_map = {nucleotide: i for i, nucleotide in enumerate(self.unique_nucleotides)}
        print("Nucleotide mapping:")
        for nucleotide, index in self.nucleotide_map.items():
            print(f"{nucleotide}: {index}")

        # Check if all sequences are of the same length
        self.sequence_length = len(sequences[0])
        if not all(len(seq) == self.sequence_length for seq in sequences):
            raise ValueError("Sequences must be the same length.")

        # One-hot encode the sequences
        encoded_sequences = []
        for sequence in sequences:
            encoded_sequence = np.zeros((len(sequence), len(self.unique_nucleotides)), dtype=np.uint8)
            for i, nucleotide in enumerate(sequence):
                encoded_sequence[i, self.nucleotide_map[nucleotide]] = 1
            encoded_sequences.append(encoded_sequence)
        
        return encoded_sequences


    def one_hot_decode(self, predictions):
        '''
        Converts model predictions (probabilities) to virus sequences.

        Parameters:
        - predictions (numpy.ndarray): Array of model predictions (probabilities).
                                Shape: (num_samples, sequence_length, num_classes).

        Returns:
        - sequences (list): List of virus nucleotides.
        '''
        
        if self.unique_nucleotides is None or self.nucleotide_map is None or self.sequence_length is None:
            raise ValueError("Encoding information is not available. Please encode sequences first.")

        num_samples, sequence_length, num_classes = predictions.shape

        if len(self.unique_nucleotides) != num_classes:
            raise ValueError("Number of classes in predictions does not match encoding information.")

        sequences = []
        for i in range(num_samples):
            sequence = ''.join(self.unique_nucleotides[np.argmax(predictions[i, j])] for j in range(sequence_length))
            sequences.append(sequence)

        return sequences