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
    for norms_id, group_test_data in sanitized_test_data.groupby("norms_id"):
        # get norms
        test_norms = test_all_norms[test_all_norms["norms_id"] == norms_id] if norms_id != UNAVAILABLE_NORMS else pd.DataFrame()
        # init scorer
        scorer = Scorer(test_specs, test_norms, group_test_data) # type: ignore
        # add score
        test_results = pd.concat([ test_results, scorer.score() ])
    # expand dict-like columns
    final_df = expand_dict_like_columns(test_results, "_std")
    # determine path of results data file
    test_results_filepath= filer.get_base_folderpath("xerox") / f"{Path(test_data_filename).stem}_scored.csv" # type: ignore
    # store final df
    final_df.to_csv(test_results_filepath, index=False)
# on error
except Exception as e:
    # notify error message
    print(e)
    # notify traceback
    TracebackNotifier(e).notify_traceback()
