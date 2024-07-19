import argparse
import pandas as pd

from pathlib import Path
from lib.Filer import Filer, TESTS_PATH
from lib.Loader import Loader
from lib.Sanitizer import Sanitizer, UNAVAILABLE_NORMS
from lib.Scorer import Scorer
from lib.Errors import TracebackNotifier
from lib.Utils import expand_dict_like_columns
# available tests list
available_tests = [ f.name for f in TESTS_PATH.glob("[!.]*") if f.is_dir ]

# argparse
parser = argparse.ArgumentParser(prog="Scoring Machine")
parser.add_argument("-t", "--test", required=True, choices=available_tests)
parser.add_argument("-e", "--expand_norms", choices=["0", "1"], default="0")
args = parser.parse_args()

try:
    # determine filename of test data file
    test_data_filename = f"data_{args.test}.csv"
    # init Filer
    filer = Filer()
    # init Loader
    loader = Loader(filer)
    # load test assets
    test_specs, test_all_norms = loader.load_test_specifications_and_norms(args.test)
    # load data to score
    test_data = loader.load_test_data(test_data_filename)
    # init Sanitizer
    sanitizer = Sanitizer(test_specs, test_data)
    # sanitize data
    sanitized_test_data = sanitizer.sanitize()
    # init test results
    test_results = pd.DataFrame()
    # loop through test data grouped by norms id
    for norms_ids, group_test_data in sanitized_test_data.groupby("norms_id", sort=False):
        # split norm_ids
        norms_list = norms_ids.split(" ") # type: ignore
        # get norms
        test_norms = test_all_norms[test_all_norms["norms_id"].isin(norms_list)] if norms_list[0] != UNAVAILABLE_NORMS else pd.DataFrame()
        # init scorer
        scorer = Scorer(test_specs, test_norms, group_test_data) # type: ignore
        # add score
        test_results = pd.concat([ test_results, scorer.score() ])
    # reset index
    test_results = test_results.reset_index()
    # if dict-like columns should be expanded
    if args.expand_norms == "1":
        # expand dict-like columns
        test_results = expand_dict_like_columns(test_results, "std_")
    # rebuild original index
    test_results = test_results.set_index("index").sort_index()
    # determine path of results data file
    test_results_filepath= filer.get_base_folderpath("xerox") / f"{Path(test_data_filename).stem}_scored.csv" # type: ignore
    # store results data
    test_results.to_csv(test_results_filepath, index=False)
# on error
except Exception as e:
    # notify error message
    print(e)
    # notify traceback
    TracebackNotifier(e).notify_traceback()
