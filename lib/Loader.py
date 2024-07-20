import json
import pandas as pd
from functools import reduce

from typing import Any
from lib.Filer import Filer
from lib.Errors import NotFoundError

class TestSpecs():

    def __init__(self, data: dict) -> None:
        self.data = data

    def get_spec(self, path: str) -> Any:
        # split json path
        path_bits = path.split(".")
        # return requested value traversing path bits
        return reduce(lambda acc, itr: acc.get(itr, {}), path_bits, self.data)

class Loader(object):

    def __init__(self, filer: Filer) -> None:
        self.filer = filer

    def load_test_specifications_and_norms(self, test: str) -> tuple[TestSpecs, pd.DataFrame]:
        # determine test foldepath
        test_folderpath = self.filer.get_test_folderpath(test)
        # determine test specifications json file
        test_specs_filepath = test_folderpath / f"{test}_specs.json"
        # init test specifications json object
        test_specs_json = {}
        # if test specifications json file exists
        if test_specs_filepath.exists():
            # open file
            with test_specs_filepath.open() as fin:
                # load test specifications into json object
                test_specs_json = json.load(fin)
        # init TestSpecs
        test_specs = TestSpecs(test_specs_json)
        # get all norms
        norms_filepath = test_folderpath / f"{test}_norms.csv"
        # init norms
        test_all_norms = pd.DataFrame()
        # if test norms file exists
        if norms_filepath.exists():
            # read norms file
            test_all_norms = pd.read_csv(norms_filepath)
        # return test specifcations and norms
        return test_specs, test_all_norms

    def load_test_data(self, data_filename: str) -> pd.DataFrame:
        # determine data filepath
        data_filepath = self.filer.get_base_folderpath("data") / data_filename # type: ignore
        # init test data variable
        data = pd.DataFrame()
        # if test data filepath exists
        if data_filepath.exists():
            # load test data
            data = pd.read_csv(data_filepath)
        # return test data
        return data
