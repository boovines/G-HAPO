import json
import random

# Load original
with open("../datasets/train_samples_math_2000.json", "r") as f:
    data = json.load(f)

# Sample 100 (Seed for reproducibility)
random.seed(42)
subset = random.sample(data, 100)

# Save
with open("datasets/train_samples_math_100.json", "w") as f:
    json.dump(subset, f)

print(f"Created ultra-light subset with {len(subset)} examples.")