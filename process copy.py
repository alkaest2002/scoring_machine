import json
import pandas as pd
import numpy as np

from pathlib import Path
from lib.Sanitizer import Sanitizer
from lib.Scorer import Scorer


def add_suffix(data, suffix):
    data.columns = [f"{c}_{suffix}" for c in data.columns]
    return data

test_folder_path = Path("./tests")
test = "demo"

# test specs
with open(test_folder_path / test / f"{test}_specs.json" ) as f:
    test_specs = json.load(f)

# test norms
norms_path = test_folder_path / test / f"{test}_norms_ita.csv"
norms = pd.read_csv(norms_path)

# load data
data = pd.read_csv(f"data_{test}.csv")
data.head()

# init sanitizer
sanitizer = Sanitizer(test_specs)

items = data.iloc[:,1:]
items = sanitizer.ensure_numeric(items)
sanitizer.check_length(items)

# init scorer
scorer = Scorer(test_specs)

for norms_id, group_data in data.groupby("norms_id"):
    group_norms = norms[norms["norms_id"] == norms_id]
    group_answers = group_data.iloc[:,1:]
    raw_scores = scorer.compute_raw_scores_compensate_for_missing_items(group_answers)
    missing_by_scale = scorer.count_missing_items_by_scale(group_answers)
    standardized_scores = scorer.compute_standard_scores(raw_scores, group_norms, "tscore")
    output = pd.concat([
        data.loc[:, "norms_id"],
        add_suffix(raw_scores, "raw"),
        add_suffix(standardized_scores, "std"),
        add_suffix(missing_by_scale, "miss"),
    ], axis=1).to_csv("./results.csv")
