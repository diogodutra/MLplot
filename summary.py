import pandas as pd


def summarize_index(df, index, drop=['count',]):
    """
    Reduces a MultiIndex DataFrame by one of its index.

    Args:
        df (MultiIndex Pandas): DataFrame with some columns as signals
        index (string): name of the index to be summarized
        drop (list, optional): functions to be discarded

    Returns:
        DataFrame: summary of 'df' by 'index' 
    """
    
    remaining_index = list(set(df.index.names) - set([index]))
    df_summarized = df.groupby(remaining_index).describe()

    if drop is not None:
        drop_level = df_summarized.columns.nlevels - 1
        df_summarized.drop(columns=drop, level=drop_level, inplace=True)

    # add the index name into the new multiindex column
    col_names = list(df_summarized.columns.names)
    col_names[-1] = index
    df_summarized.columns.names = col_names

    return df_summarized


def summarize(df, keep=[], drop=['count',], verbose=False):
    """
    Reduces a MultiIndex DataFrame by all its indices.

    Args:
        df (MultiIndex Pandas): DataFrame with some columns as signals
        keep (list of strings): names of the indices to not be summarized
        drop (list of strings, optional): functions to be discarded
        verbose (bool, optional): print steps on console

    Returns:
        DataFrame: summary of 'df' by all its indices (except those in 'keep')
    """

    indices = list(set(df.index.names) - set(keep))
    df_summarized = df
    for index in indices:
        if verbose: print(f'\rSummarizing {index}', end=" ")
        df_summarized = summarize_index(df_summarized, index, drop)

    if verbose: print(f'\rSummarized {len(indices)} indices.')

    return df_summarized