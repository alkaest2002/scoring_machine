import pandas as pd

def expand_dict_like_columns(df: pd.DataFrame, regex_for_dict_like: str) -> pd.DataFrame:
    # filter dict-like columns based on reges
    dict_like_columns = df.filter(regex=regex_for_dict_like)
    # filter all columns except dict-like
    df_except_dictlike = df.loc[:, ~(df.columns.isin(dict_like_columns.columns))]
    # init final dataset
    final_df = df_except_dictlike
    # expand dict-like columns
    for col_dict_name, col_dict in dict_like_columns.items():
        # concatenate
        final_df = pd.concat([
            final_df,
            pd.json_normalize(col_dict).add_prefix(f"{col_dict_name}_") # type: ignore
        ], axis=1)
    # return final df
    return final_df
