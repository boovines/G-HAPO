import pandas as pd
import json
import numpy as np

# Load datasets
# s1k
s1k_df = pd.read_json("../datasets/s1k_answers_subset.json", lines=True)

# reclor (val.json)
reclor_df = pd.read_json("../datasets/ReClor/val.json")

# hapo math (train_samples_math_100.json)
hapo_df = pd.read_json("../datasets/train_samples_math_100.json")

# Set random seed for reproducibility
np.random.seed(42)

# Function to shuffle dataframe
def shuffle_df(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# 1. Process HAPO Math
# Need: 34 Train, 9 Val
hapo_shuffled = shuffle_df(hapo_df)
hapo_train = hapo_shuffled.iloc[:34].copy()
hapo_val = hapo_shuffled.iloc[34:43].copy()
# Label source for tracking
hapo_train['dataset_source'] = 'HAPO'
hapo_val['dataset_source'] = 'HAPO'

# 2. Process ReClor
# Need: 33 Train, 8 Val, 50 Test
reclor_shuffled = shuffle_df(reclor_df)
reclor_train = reclor_shuffled.iloc[:33].copy()
reclor_val = reclor_shuffled.iloc[33:41].copy()
reclor_test = reclor_shuffled.iloc[41:91].copy()
# Label source
reclor_train['dataset_source'] = 'ReClor'
reclor_val['dataset_source'] = 'ReClor'
reclor_test['dataset_source'] = 'ReClor'

# 3. Process s1K
# Need: Train(33), Val(8), Test(50)
# Break down by category
s1k_math = shuffle_df(s1k_df[s1k_df['cot_type'] == 'math'])
s1k_sci = shuffle_df(s1k_df[s1k_df['cot_type'] == 'science'])
s1k_cross = shuffle_df(s1k_df[s1k_df['cot_type'] == 'crossword'])

# Train allocations: 5 Cross, 14 Math, 14 Sci
s1k_train_c = s1k_cross.iloc[:5]
s1k_train_m = s1k_math.iloc[:14]
s1k_train_s = s1k_sci.iloc[:14]
s1k_train = pd.concat([s1k_train_c, s1k_train_m, s1k_train_s])

# Val allocations: 2 Cross, 3 Math, 3 Sci
s1k_val_c = s1k_cross.iloc[5:7]
s1k_val_m = s1k_math.iloc[14:17]
s1k_val_s = s1k_sci.iloc[14:17]
s1k_val = pd.concat([s1k_val_c, s1k_val_m, s1k_val_s])

# Test allocations: 8 Cross (Remaining), 21 Math, 21 Sci
s1k_test_c = s1k_cross.iloc[7:] # Should be 8
s1k_test_m = s1k_math.iloc[17:38]
s1k_test_s = s1k_sci.iloc[17:38]
s1k_test = pd.concat([s1k_test_c, s1k_test_m, s1k_test_s])

s1k_train['dataset_source'] = 's1K'
s1k_val['dataset_source'] = 's1K'
s1k_test['dataset_source'] = 's1K'

# 4. Merge Validations and Trains
final_train = pd.concat([hapo_train, reclor_train, s1k_train]).sample(frac=1, random_state=42).reset_index(drop=True)
final_val = pd.concat([hapo_val, reclor_val, s1k_val]).sample(frac=1, random_state=42).reset_index(drop=True)
final_test_reclor = reclor_test.reset_index(drop=True)
final_test_s1k = s1k_test.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save to JSON
final_train.to_json("merged_train.json", orient='records', lines=True)
final_val.to_json("merged_val.json", orient='records', lines=True)
final_test_reclor.to_json("test_reclor.json", orient='records', lines=True)
final_test_s1k.to_json("test_s1k.json", orient='records', lines=True)

# Print Summary
print("Final Dataset Splits:")
print(f"Train Set Size: {len(final_train)}")
print(final_train['dataset_source'].value_counts())
print("\nValidation Set Size: {len(final_val)}")
print(final_val['dataset_source'].value_counts())
print("\nTest Set ReClor Size: {len(final_test_reclor)}")
print("\nTest Set s1K Size: {len(final_test_s1k)}")
print(final_test_s1k['cot_type'].value_counts())