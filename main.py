import pandas as pd

splits = {'train': 'plain_text/train-00000-of-00001.parquet', 
          'test': 'plain_text/test-00000-of-00001.parquet', 
          'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}

df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

# Create balanced sample of 25 rows from each label
sample_size_per_label = 25
balanced_df = pd.concat([
    df[df['label'] == 0].sample(n=sample_size_per_label, random_state=42),
    df[df['label'] == 1].sample(n=sample_size_per_label, random_state=42)
])

# Shuffle the rows to mix the labels
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Shape of balanced dataset: {balanced_df.shape}")
print("\nLabel distribution:")
print(balanced_df['label'].value_counts())

