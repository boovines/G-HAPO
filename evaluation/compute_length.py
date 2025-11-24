import pandas as pd
import polars as pl

from transformers import AutoTokenizer

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the parquet file produced by lighteval for a specific evaluation run.")
parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer to use for length calculation")
args = parser.parse_args()
path = args.path

try:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
except Exception:
    # Fallback if specific tokenizer fails (e.g. invalid repo)
    print(f"Warning: Could not load {args.tokenizer}, falling back to Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lens = []

if "lcb:codegeneration" in path:
    df_parquet = pl.read_parquet(path)
    for i in range(len(df_parquet)):
        assert len(df_parquet[i]["predictions"]) == 1
        for pred in df_parquet[i]["predictions"][0]:
            lens.append(len(tokenizer.encode(pred, add_special_tokens=False)))
else:
    df_parquet = pd.read_parquet(path)
    for index, row in df_parquet.iterrows():
        for pred in row["predictions"]:
            lens.append(len(tokenizer.encode(pred, add_special_tokens=False)))


print("Response length: {}\n\n".format(path, sum(lens)/len(lens)))