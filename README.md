
# OpenAI Evaluation Tool Tutorial

[![Tutorial Video](https://img.youtube.com/vi/pgyhq-WagIg/0.jpg)](https://youtu.be/pgyhq-WagIg)

This repository contains the sample code and data used in the video tutorial demonstrating OpenAI's evaluation tool.

## Repository Structure

- `imdb_sample.csv`: Sample dataset containing movie reviews from IMDB, used for sentiment analysis evaluation
- `prompts.md`: Contains the system and user prompts used to generate responses from OpenAI
- `main.py`: Main script that processes the IMDB dataset and prepares it for evaluation

## Data Source

The sample dataset is a balanced subset of the IMDB movie reviews, containing:
- 25 negative reviews (label: 0)
- 25 positive reviews (label: 1)

## Evaluation Setup

The evaluation uses a simple binary classification task where:
- Positive reviews should output 1
- Negative reviews should output 0

This setup allows for clear measurement of the model's ability to understand and classify movie review sentiments.
