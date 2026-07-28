import os

import pandas as pd
from dotenv import load_dotenv
import kagglehub
import shutil
from pathlib import Path

def clear_checkpoints():
    path = "../models/Faster_RCNN/checkpoints/"
    for file in os.listdir(path):
        os.remove(path + file)

def merge_csvs(files: list[str], output: str) -> None:
    """
    Merges multiple CSV files into a single output file. Each input file is read
    and concatenated, and the combined output is saved into the specified output
    file. The operation only proceeds if two or more input files are provided.

    :param files: A list of file paths to the input CSV files to be merged.
    :param output: The file path where the merged CSV content should be saved.
    :return: None
    """

    # Do nothing if 0 or 1 files are provided
    if len(files) < 2:
        return
    dfs = [pd.read_csv(f) for f in files]
    pd.concat(dfs).to_csv(output, index=False)
    
def download_dataset():
    # Get Kaggle credentials from .env
    load_dotenv()
    kaggle_username, kaggle_key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    
    # Set credentials in os.environ so Kaggle API can find them automatically
    if kaggle_username:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        os.environ["KAGGLE_KEY"] = kaggle_key
    
    if not kaggle_key:
        print("Warning: KAGGLE_KEY not found. Please set it as an environment variable.")
    
    # Download dataset from Kaggle if not already downloaded
    path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
    
    print("Path to dataset files:", path)
    
    source_dir = Path(path)
    target_dir = Path('../data/cbis-ddsm')
    
    # Ensure the target directory exists and is empty
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if len(os.listdir(target_dir)) == 0:
        shutil.copytree(str(source_dir), str(target_dir),
                        dirs_exist_ok=True)  # Copy the data to the target directory instead of moving it
        print("Copied data to target location")
        
        # Merge Mass CSVs into a single file
        merge_csvs(['../data/cbis-ddsm/csv/mass_case_description_test_set.csv',
                    '../data/cbis-ddsm/csv/mass_case_description_train_set.csv'],
                   '../data/cbis-ddsm/csv/mass_case_description_merged.csv')
    else:
        print("Target directory is not empty. Skipping data copy.")
        
        
        
if __name__ == "__main__":
    clear_checkpoints()