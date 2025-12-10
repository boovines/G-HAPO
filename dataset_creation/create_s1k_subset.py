# from datasets import load_dataset

# # The dataset name is "simplescaling/s1K"
# dataset_name = "simplescaling/s1K"

# # Download and load the dataset
# # This will return a DatasetDict containing the available splits (e.g., 'train')
# s1k_dataset = load_dataset(dataset_name)

# # Access the training split
# train_data = s1k_dataset['train']

# # Print the first example to verify
# print(train_data[0])

# # You can also save the dataset locally to a specific file format (e.g., CSV or JSON) if needed:
# train_data.to_csv("./datasets/s1k_train.csv", index=False)
import pandas as pd
import re
import os

# 1. Load the dataset
df = pd.read_csv("../datasets/s1k_train.csv")

# 2. Define regex patterns for explicit answer markers and proof markers
patterns = {
    'Boxed': r'\\boxed\{',
    'Answer Colon': r'(?i)\banswer:\s*',        # Case insensitive "Answer:"
    'Final Answer Is': r'(?i)the final answer is', # Case insensitive phrase
    "Number": r"^[+-]?(\d+(\.\d*)?|\.\d+)$"
    # 'QED': r'\\blacksquare|\\square'            # Proof end markers
}

# 3. Create boolean masks for each pattern
has_boxed = df['solution'].str.contains(patterns['Boxed'], regex=True, na=False)
has_answer_colon = df['solution'].str.contains(patterns['Answer Colon'], regex=True, na=False)
has_final_phrase = df['solution'].str.contains(patterns['Final Answer Is'], regex=True, na=False)
# has_number = df['solution'].str.contains(patterns['Number'], regex=True, na=False)
is_short = df['solution'].str.len() < 30
# has_qed = df['solution'].str.contains(patterns['QED'], regex=True, na=False)

# 4. Identify "Explicit Answer" presence
# Matches any solution containing "\boxed{}", "Answer:", or "The final answer is"
has_explicit_answer =is_short | has_boxed | has_answer_colon | has_final_phrase 


# 7. Create the subset DataFrame
subset_df = df[has_explicit_answer].copy()

# 8. Save the subset to CSV and JSON
csv_filename = "s1k_answers_subset.csv"
json_filename = "s1k_answers_subset.json"

subset_df.to_csv(f"./datasets/{csv_filename}", index=False)
subset_df.to_json(f"./datasets/{json_filename}", orient='records', lines=True)

# 9. Verification Output
print(f"Total Original Solutions: {len(df)}")
print(f"Included in Subset: {len(subset_df)}")
print(f"Subset Proportion: {len(subset_df) / len(df):.3f}")
print(f"\nFiles generated: {csv_filename}, {json_filename}")