import numpy as np
import pandas as pd
from functools import cached_property

class Scorer():

    def __init__(self, test_specs: dict, test_norms: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.test_specs = test_specs
        self.test_norms = test_norms
        self.test_data = test_data
        self.straight_items_by_scale, self.reversed_items_by_scale = self.convert_to_matrices()

    @cached_property
    def scale_names(self) -> list[str]:
        return [ scale[0] for scale in self.test_specs.get_spec("scales") ] # type: ignore

    @cached_property
    def norms_answers(self) -> pd.DataFrame:
        return self.test_data.loc[:, ["norms_id"]]

    @cached_property
    def item_answers(self) -> pd.DataFrame:
        return self.test_data.drop(columns=["norms_id"])

    def convert_to_matrices(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # init matrices to be returned
        items_by_scale = pd.DataFrame()
        reversed_items_by_scale = pd.DataFrame()
        # iterate over scales
        for scale in self.test_specs.get_spec("scales"): # type: ignore
            # get current scale label, straight and reversed items
            scale_label, straight_items_indices, reversed_items_indices = scale
            # iterate over type of items (either straight or reversed)
            for df, items_indices in [
                (items_by_scale, straight_items_indices),
                (reversed_items_by_scale, reversed_items_indices)
            ]:
                # init items series with zeroes
                items = pd.Series(np.zeros(self.test_specs.get_spec("length"))) # type: ignore
                # correct item indices, since items are 1-based while matrices are 0-based
                matrix_indices = pd.Series(items_indices).sub(1)
                # "switch to 1" items belonging to current scale
                items[matrix_indices] = 1
                # add current scale items matrix to approriate df
                df[scale_label] = items
        # return stright/reversed items matrices
        return items_by_scale, reversed_items_by_scale

    def compute_raw_score(self, items_by_scale: pd.DataFrame, fillna_value: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        # clone item answers
        item_answers = self.item_answers.copy()
        # compute raw scores
        raw_scores = np.dot(item_answers.fillna(fillna_value).sub(fillna_value).abs(), items_by_scale)
        # intercept numpy errors
        with np.errstate(divide='ignore', invalid='ignore'):
            # compute mean raw scores
            mean_scores = np.true_divide(raw_scores, items_by_scale.sum(axis=0).to_numpy())
            # replace NaNs, Infs with 0
            mean_scores = np.nan_to_num(mean_scores, nan=0.0, posinf=0.0, neginf=0.0)
        # return matrices as pandas dataframes
        return (
            pd.DataFrame(raw_scores, index=self.item_answers.index, columns=self.scale_names), # type: ignore
            pd.DataFrame(mean_scores, index=self.item_answers.index, columns=self.scale_names) # type: ignore
        )

    def compute_raw_scores_components(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # set fillna value for straight items
        fillna_value = 0
        # compute scores
        sum_straight, mean_straight = self.compute_raw_score(self.straight_items_by_scale, fillna_value)
        # set fillna value for reverse items
        fillna_value = sum(self.test_specs.get_spec("likert").values()) # type: ignore
        # compute reversed items
        sum_reversed, mean_reversed = self.compute_raw_score(self.reversed_items_by_scale, fillna_value)
        # return results
        return sum_straight, sum_reversed, mean_straight, mean_reversed

    def count_items_by_scale(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        return self.straight_items_by_scale.sum(), self.reversed_items_by_scale.sum()

    def count_missing_items_by_scale(self) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        # lambda fn
        lambda_fn = lambda x: self.item_answers.T.loc[x.astype(bool).values].isna().sum()
        # missing straight items per scale
        sum_of_missing_straight_items_by_scale = self.straight_items_by_scale.apply(lambda_fn)
        # missing reversed items per scale
        sum_of_missing_reversed_items_by_scale = self.reversed_items_by_scale.apply(lambda_fn)
        # if we want results splitted by straight/reversed items
        return sum_of_missing_straight_items_by_scale, sum_of_missing_reversed_items_by_scale

    def compute_raw_scores_compensate_for_missing_items(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # get splitted data that is needed for computing raw scores while compensating for missing items
        sum_straight, sum_reversed, mean_straight, mean_reversed = self.compute_raw_scores_components()
        missing_straight, missing_reversed = self.count_missing_items_by_scale()
        items_straight, items_reversed = self.count_items_by_scale()
        # init list
        raw_components = []
        # compute corrected raw scores
        for sum_scores, missing_items_by_scale, items_by_scale in [
            (sum_straight, missing_straight, items_straight),
            (sum_reversed, missing_reversed, items_reversed)
        ]:
            # compute how many items where effectively responded (by scale)
            items_with_answers_by_scale = items_by_scale - missing_items_by_scale
            # intercept numpy errors
            with np.errstate(divide='ignore', invalid='ignore'):
                # compute mean responses
                mean_results = np.true_divide(sum_scores, items_with_answers_by_scale)
                # replace NaNs, Infs with 0
                mean_results = np.nan_to_num(mean_results, nan=0.0, posinf=0.0, neginf=0.0)
            # compute corrected results
            corrected_results = mean_results * items_by_scale.T.to_numpy()
            # append corrected results to list
            raw_components.append(corrected_results)
        # assemble raw scores dataframe
        raw_scores = pd.DataFrame(sum(raw_components), index=self.item_answers.index, columns=self.scale_names).astype(int)
        # return results
        return (
            raw_scores.astype(int),
            (sum_straight+sum_reversed).div(items_straight.sub(missing_straight)+items_reversed.sub(missing_reversed)).round(2)
        )

    def compute_standard_scores(self, raw_scores, norms: pd.DataFrame, norms_col: str) -> pd.DataFrame:
        # if norms is an empty dataframe
        if norms.empty:
            # return empty dataframe with the same structure of raw_scores
            return pd.DataFrame().reindex_like(raw_scores) # type: ignore
        # function to take standard scores
        def get_standard_scores(series, **kwargs):
            # get kwargs
            norms, norms_col = kwargs["norms"], kwargs["norms_col"]
            # determine wich norms to use for current scale
            norms_to_use = norms[norms["scale"].eq(series.name)]
            # get standard scores
            stds = pd.merge_asof(series.to_frame().sort_values(by=series.name), norms_to_use, left_on=series.name, right_on="raw")
            # return form fifth column onwards
            return stds.iloc[:, 4:].to_dict(orient="records")
        # return standard scores
        return raw_scores.apply(get_standard_scores, norms=norms, norms_col=norms_col)

    def score(self, type_of_norms: str = "std"):
        # compute raw scores for each scale
        raw_scores, mean_scores = self.compute_raw_scores_compensate_for_missing_items()
        # compute missing items for each scale
        missing_by_scale = sum(self.count_missing_items_by_scale())
        # compute std scores for each scale
        standardized_scores = self.compute_standard_scores(raw_scores, self.test_norms, type_of_norms)
        # return results
        return pd.concat([
            self.norms_answers,
            self.item_answers,
            missing_by_scale.add_suffix("_miss"), #type: ignore
            raw_scores.add_suffix("_raw"),
            mean_scores.add_suffix("_mean"),
            standardized_scores.add_suffix(f"_{type_of_norms}"),
        ], axis=1)
