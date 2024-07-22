import numpy as np
import pandas as pd

from functools import cached_property
from lib.Errors import ValidationError
from lib.Loader import TestSpecs

UNAVAILABLE_NORMS = "n.a."

class Sanitizer():

    def __init__(self, test_specs: TestSpecs, test_data: pd.DataFrame) -> None:
        self.test_specs = test_specs
        self.test_data = test_data

    @cached_property
    def norms(self) -> pd.DataFrame:
        # if norms_id is not present in dataframe columns
        if "norms_id" not in self.test_data.columns:
            # create a norms dataframe filled with UNAVAILABLE_NORMS constant
            return pd.DataFrame({ "norms_id": UNAVAILABLE_NORMS }, index=self.test_data.index)
        # otherwiese return norms_id columns
        return self.test_data.loc[:, ["norms_id"]]

    @cached_property
    def item_answers(self) -> pd.DataFrame:
        # return items answers after dropping norms_id columns
        return self.test_data.drop(columns=["norms_id"], errors="ignore")

    def sanitize_norms(self) -> pd.DataFrame:
        # get available norms as a set
        available_norms = set(self.test_specs.get_spec("norms"))
        # set condition to check
        condition = self.norms.map(lambda x: set(x.split(" ")).issubset(available_norms)).values
        # cleanup norms
        self.norms.where(condition, UNAVAILABLE_NORMS)
        # return norms
        return self.norms

    def sanitize_item_answers(self) -> pd.DataFrame | pd.Series:
        return (
            self.item_answers
                .apply(lambda x: pd.to_numeric(x, errors="coerce", downcast="integer"))
                .clip(self.test_specs.get_spec("likert.min"), self.test_specs.get_spec("likert.max"))
        )

    def sanitize(self) -> pd.DataFrame:
        # if test data doesn't match test specificaions
        if self.norms.shape[1] + self.item_answers.shape[1] != self.test_specs.get_spec("length") + 1:
            # raise error
            raise ValidationError("Test data is not compatible with test specifications.")
        # sanitize norms and item answers
        sanitized_norms = self.sanitize_norms()
        sanitized_items_answers = self.sanitize_item_answers()
        # return
        return pd.concat([ sanitized_norms, sanitized_items_answers ], axis=1)
