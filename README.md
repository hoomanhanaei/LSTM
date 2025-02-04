[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

# RNN-based LSTM

## Overview
A Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN), is designed to efficiently process and analyze sequential data.
This model was developed to predict the next emerging variant in viral genomes, leveraging its ability to capture complex dependencies in time-series data.
By training an LSTM model on historical genomic sequences of a virus variants, and their patterns of emergence,
the model can learn to identify subtle patterns and trends to predict potential future variants.
Additionally, LSTM is capable of handling variable-length sequences, which is crucial given the variability in viral sequences across different strains and variants. 

## Features
This software processes and analyzes sequences using an **LSTM (Long Short-Term Memory) model**.
It follows a structured pipeline for **sequence alignment, padding, encoding, window generation, hyperparameter tuning, and model training**.
- Loads and process DNA sequences from **FASTA** files.
- Performs **sequence padding** for uniform lengths.
- **One-hot encode** DNA sequences (`A, C, T, G, -`).
- Generates **training windows** for model input.
- Performs **hyperparameter tuning** for optimal LSTM training.
- Trains an **LSTM model** to predict future mutations.

## Project Structure
```
📂 project_root/
├── 📂 modules/
│   ├── data_loader.py  # Load sequences from FASTA files
│   ├── sequence_padder.py  # Pad sequences to match reference length
│   ├── encoder_decoder.py  # One-hot encode DNA sequences
│   ├── window_generator.py  # Generate training windows
│   ├── encoder_decoder.py  # Encode sequences into numerical format
├── 📂 lstm_tuner/
│   ├── tuner.py  # Hyperparameter tuning
│   ├── model_builder.py  # Build LSTM model
├── 📂 model/  # model's main script
├── main.py  # Main script to run the pipeline
├── README.md  # Project documentation

```

## Usage
```
python main.py --ref_path path/to/reference.fasta --mut_path path/to/mutated.fasta --num_mut_seqs --out_dir ./output
```

## Arguments

| Argument       | Description                                      |
|---------------|--------------------------------------------------|
| `--ref_path`  | Path to reference DNA sequence (FASTA format).  |
| `--mut_path`  | Path to mutated DNA sequences (FASTA format).   |
| `--num_mut_seqs` | *(Optional)* Number of mutated sequences to process. |
| `--out_dir`   | Output directory for aligned sequences. **Default:** `./output` |
