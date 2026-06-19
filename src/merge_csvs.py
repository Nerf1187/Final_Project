import pandas as pd


def merge_csvs(files: list[str], output: str):
    if len(files) < 2:
        return
    dfs = [pd.read_csv(f) for f in files]
    pd.concat(dfs).to_csv(output, index=False)
    
    
if __name__ == "__main__":
    merge_csvs(['../data/cbis-ddsm/csv/mass_case_description_test_set.csv',
                '../data/cbis-ddsm/csv/mass_case_description_train_set.csv'],
               '../data/cbis-ddsm/csv/mass_case_description_merged.csv')