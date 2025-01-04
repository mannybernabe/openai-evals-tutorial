import pandas as pd

# Define file paths for different data splits
splits = {
    'train': 'plain_text/train-00000-of-00001.parquet',
    'test': 'plain_text/test-00000-of-00001.parquet',
    'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
}

# Load the training dataset from Hugging Face's dataset repository
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

# Create a balanced sample dataset:
# - 25 samples from negative reviews (label=0)
# - 25 samples from positive reviews (label=1)
sample_size_per_label = 25
balanced_df = pd.concat([
    df[df['label'] == 0].sample(n=sample_size_per_label, random_state=42),
    df[df['label'] == 1].sample(n=sample_size_per_label, random_state=42)
])

# Shuffle the dataset for better randomization
# - frac=1 means shuffle 100% of the data
# - random_state=42 ensures reproducible results
# - reset_index updates the row numbers sequentially after shuffling
balanced_df = (balanced_df
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))

# Export to CSV file
# - index=False prevents writing row numbers to the CSV
# - This format is compatible with Excel, Google Sheets, etc.
balanced_df.to_csv('imdb_sample.csv', index=False)

