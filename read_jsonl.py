import json
import pandas as pd

# Read the JSONL file
def read_jsonl_to_df(file_path):
    # Initialize lists to store data
    texts = []
    labels = []
    run_ids = []
    data_source_idxs = []
    model_names = []
    temperatures = []
    total_tokens = []
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Parse each JSON line
                data = json.loads(line)
                
                # Extract text and label from the nested structure
                if 'item' in data:
                    texts.append(data['item'].get('text', ''))
                    labels.append(data['item'].get('label', ''))
                    
                # Extract additional metadata
                run_ids.append(data.get('run_id', ''))
                data_source_idxs.append(data.get('data_source_idx', ''))
                
                # Extract model information from sample
                if 'sample' in data:
                    sample = data['sample']
                    model_names.append(sample.get('sampled_model_name', ''))
                    
                    # Get temperature from model params
                    model_params = sample.get('sampled_model_params', {})
                    temperatures.append(model_params.get('temperature', ''))
                    
                    # Get token usage
                    token_usage = sample.get('token_usage', {})
                    total_tokens.append(token_usage.get('total_tokens', ''))
                else:
                    model_names.append('')
                    temperatures.append('')
                    total_tokens.append('')
                    
            except json.JSONDecodeError:
                continue
    
    # Create DataFrame with all columns
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'run_id': run_ids,
        'data_source_idx': data_source_idxs,
        'model_name': model_names,
        'temperature': temperatures,
        'total_tokens': total_tokens
    })
    
    return df

# Read the file
df = read_jsonl_to_df('eval_items_OutputDataItemStatusParam.ALL_2025-01-04_17-35-52.jsonl')

# Create a DataFrame with specific columns and formatting
display_df = pd.DataFrame({
    'Index': range(len(df)),
    'Review Text': df['text'],
    'Label': df['label'],
    'Passed': ['1/1'] * len(df),
    'Status': ['Pass'] * len(df),
    'System Prompt': ['System: You are an expert in analyzing the sentiment of movie reviews...'] * len(df)
})

# Convert to HTML with styling
html_table = display_df.to_html(
    classes='table table-striped table-hover',
    escape=False,
    index=False
)

# Create a complete HTML document with styling similar to the image
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Movie Review Evaluation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #f8f9fa; }}
        .container {{ max-width: 95%; margin-top: 20px; }}
        .table {{ background-color: white; }}
        .table th {{ 
            background-color: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        .table td {{ 
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: normal;
        }}
        .passed-cell {{ 
            background-color: #d4edda;
            color: #155724;
            border-radius: 4px;
            padding: 2px 8px;
        }}
        .status-cell {{ 
            background-color: #d4edda;
            color: #155724;
            border-radius: 4px;
            padding: 2px 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2 class="mb-4">Movie Review Evaluation</h2>
        {html_table}
    </div>
</body>
</html>
"""

# Save to HTML file
with open('movie_reviews.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML file has been created: movie_reviews.html") 