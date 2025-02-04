[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)

# RNN-based LSTM

## Overview
This software processes and analyzes DNA sequences using an **LSTM (Long Short-Term Memory) model**.
It follows a structured pipeline for **sequence alignment, padding, encoding, window generation, hyperparameter tuning, and model training**.

## Features
- Load and process DNA sequences from **FASTA** files.
- Perform **sequence padding** for uniform lengths.
- **One-hot encode** DNA sequences (`A, C, T, G, -`).
- Generate **training windows** for model input.
- Perform **hyperparameter tuning** for optimal LSTM training.
- Train an **LSTM model** to predict future mutations.

## Project Structure
