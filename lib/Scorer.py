import re
import numpy as np
import pandas as pd
from functools import cached_property
from lib.Loader import TestSpecs

class Scorer():

    def __init__(self, test_specs: TestSpecs, test_norms: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.test_specs = test_specs
        self.test_norms = test_norms
        self.test_data = test_data
        self.straight_items_by_scale, self.reversed_items_by_scale = self.convert_to_matrices()

    @cached_property
    def scales(self) -> list[str]:
        return [ scale[0] for scale in self.test_specs.get_spec("scales") ]

    @cached_property
    def norms(self) -> pd.DataFrame:
        return self.test_data.loc[:, ["norms_id"]]

    @cached_property
    def answers(self) -> pd.DataFrame:
        return self.test_data.drop(columns=["norms_id"])

    @cached_property
    def count_items_by_scale(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        return self.straight_items_by_scale.sum(), self.reversed_items_by_scale.sum()

    @cached_property
    def missing_items_by_scale(self) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        # lambda fn
        lambda_fn = lambda x: self.answers.T.loc[x.astype(bool).values].isna().sum()
        # missing straight items per scale
        sum_of_missing_straight_items_by_scale = self.straight_items_by_scale.apply(lambda_fn)
        # missing reversed items per scale
        sum_of_missing_reversed_items_by_scale = self.reversed_items_by_scale.apply(lambda_fn)
        # if we want results splitted by straight/reversed items
        return sum_of_missing_straight_items_by_scale, sum_of_missing_reversed_items_by_scale

    def convert_to_matrices(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # init matrices to be returned
        items_by_scale = pd.DataFrame()
        reversed_items_by_scale = pd.DataFrame()
        # iterate over scales
        for scale in self.test_specs.get_spec("scales"):
            # get current scale label, straight and reversed items
            scale_label, straight_items_indices, reversed_items_indices = scale
            # iterate over type of items (either straight or reversed)
            for df, items_indices in [
                (items_by_scale, straight_items_indices),
                (reversed_items_by_scale, reversed_items_indices)
            ]:
                # init items series with zeroes
                items = pd.Series(np.zeros(self.test_specs.get_spec("length")))
                # correct item indices, since items are 1-based while matrices are 0-based
                matrix_indices = pd.Series(items_indices).sub(1)
                # "switch to 1" items belonging to current scale
                items[matrix_indices] = 1
                # add current scale items matrix to approriate dataframe
                df[scale_label] = items
        # return stright/reversed items matrices
        return items_by_scale, reversed_items_by_scale

    def compute_raw_score_component(self, items_by_scale: pd.DataFrame, fillna_value: int) -> pd.DataFrame:
        # clone item answers
        answers = self.answers.copy()
        # compute raw scores
        raw_scores = np.dot(answers.fillna(fillna_value).sub(fillna_value).abs(), items_by_scale)
        # return matrices as pandas dataframes
        return pd.DataFrame(raw_scores, index=self.answers.index, columns=self.scales) # type: ignore

    def compute_raw_scores(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # set fillna value for straight items
        fillna_value = 0
        # compute scores
        raw_scores_straight = self.compute_raw_score_component(self.straight_items_by_scale, fillna_value)
        # set fillna value for reverse items
        fillna_value = sum(self.test_specs.get_spec("likert").values())
        # compute reversed items
        raw_scores_reversed = self.compute_raw_score_component(self.reversed_items_by_scale, fillna_value)
        # return results
        return raw_scores_straight, raw_scores_reversed

    def compute_scores(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # intercept numpy errors
        with np.errstate(divide='ignore', invalid='ignore'):
            # get items by scale and type of item
            count_items_straight, count_items_reversed = self.count_items_by_scale
            # get missing items by scale and type of item
            missing_items_straight, missing_items_reversed = self.missing_items_by_scale
            # get raw scores by scale
            raw_scores_straight, raw_scores_reversed = self.compute_raw_scores()
            # init list of components for computing corrected raw scores (i.e., take into account missing items)
            corrected_raw_components = []
            # compute corrected raw scores
            for sum_scores, missing_items_by_scale, count_items_by_scale in [
                (raw_scores_straight, missing_items_straight, count_items_straight),
                (raw_scores_reversed, missing_items_reversed, count_items_reversed)
            ]:
                # compute how many items where effectively responded (by scale)
                items_with_answers_by_scale = missing_items_by_scale.rsub(count_items_by_scale)
                # compute mean responses
                mean_results = np.true_divide(sum_scores, items_with_answers_by_scale)
                # replace NaNs, Infs with 0
                mean_results = np.nan_to_num(mean_results, nan=0, posinf=0, neginf=0)
                # compute corrected results
                corrected_results = mean_results * count_items_by_scale.to_numpy()
                # append corrected results to list
                corrected_raw_components.append(corrected_results)
            # assemble raw scores dataframe
            corrected_raw_scores = pd.DataFrame(sum(corrected_raw_components), index=self.answers.index, columns=self.scales)
            # compute mean scores
            mean_scores = (
                raw_scores_straight + raw_scores_reversed).div(count_items_straight.sub(missing_items_straight) + count_items_reversed.sub(missing_items_reversed)
            )
            # replace NaNs, Infs with 0
            mean_scores = np.nan_to_num(mean_scores, nan=np.nan, posinf=np.nan, neginf=np.nan)
            # convert mean_scores to dataframe
            mean_scores = pd.DataFrame(mean_scores, index=self.answers.index, columns=self.scales)
            # return results
            return raw_scores_straight+raw_scores_reversed, corrected_raw_scores.astype(int), mean_scores.round(2)

    def compute_standard_scores(self, raw_scores, norms: pd.DataFrame, norms_col: str) -> pd.DataFrame:
        # if norms is an empty dataframe
        if norms.empty:
            # return empty dataframe with the same structure of raw_scores
            return pd.DataFrame().reindex_like(raw_scores)
        # function to take standard scores
        def get_standard_scores(series, **kwargs):
            # get kwargs
            norms, norms_col = kwargs["norms"], kwargs["norms_col"]
            # prepare norms table
            # need to pivto in case user requested multiple norms
            norms = norms.pivot_table(index=['scale','raw'], columns=['norms_id'], values=['std']).reset_index()
            # flatten multilevel columns index
            norms = pd.DataFrame(norms.to_records())
            # clean up columns labels
            norms.columns = [ "_".join(re.findall(r'\b\w+\b', c))  for c in norms.columns ]
            # drop unwnated column
            norms = norms.drop(columns="index")
            # determine wich norms to use for current scale
            norms_to_use = norms[norms["scale"].eq(series.name)]
            # get standard scores
            stds = pd.merge_asof(
                series.to_frame().reset_index().sort_values(by=series.name),
                norms_to_use.sort_values(by="raw"), # type: ignore
                left_on=series.name,
                right_on="raw",
                direction="nearest"
            )
            # reindex to re-establish original order of series
            stds = stds.set_index("index").sort_index()
            # return form fifth column onwards
            return stds.iloc[:, 3:].add_suffix(f"_{norms.iloc[0,0]}").to_dict(orient="records")
        # return standard scores
        return raw_scores.apply(get_standard_scores, norms=norms, norms_col=norms_col)

    def score(self, type_of_norms: str = "std"):
        # compute missing items for each scale
        missing_by_scale = sum(self.missing_items_by_scale)
        # compute raw scores for each scale
        raw_scores, corrected_raw_scores, mean_scores = self.compute_scores()
        # compute std scores for each scale
        standardized_scores = self.compute_standard_scores(corrected_raw_scores, self.test_norms, type_of_norms)
        # return results
        return pd.concat([
            self.norms,
            self.answers,
            missing_by_scale.add_prefix("missing_"), # type: ignore
            raw_scores.add_prefix("raw_"),
            corrected_raw_scores.add_prefix("corrected_raw_"),
            mean_scores.add_prefix("mean_"),
            standardized_scores.add_prefix(f"{type_of_norms}_"),
        ], axis=1)
