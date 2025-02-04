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

