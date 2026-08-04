import os

import pandas as pd
from dotenv import load_dotenv
import kagglehub
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

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
    
    # Move the dataset to local directory if needed
    
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

def create_dataframes():
    # Full dataframe
    dicom_df = pd.read_csv('../data/cbis-ddsm/csv/dicom_info.csv')
    mass_df = pd.read_csv('../data/cbis-ddsm/csv/mass_case_description_merged.csv')
    
    mass_df['uid'] = mass_df['image file path'].str.split('/').str[0] + '_' + mass_df['abnormality id'].astype(str)
    mass_df['pathology'] = mass_df['pathology'].str.split('_').str[0] # Merge "BENIGN" and "BENIGN WITHOUT CALLBACK" into a single value
    mass_df = mass_df[['uid', 'pathology']]
    
    # Change image path to relative path
    dicom_df['image_path'] = dicom_df['image_path'].apply(lambda x: x.replace('CBIS-DDSM/', '../data/cbis-ddsm/'))
    
    # Remove calcification images. Only focusing on masses for now
    dicom_df = dicom_df[~dicom_df['PatientID'].str.contains('Calc')]
    
    dicom_df.reset_index(inplace=True, drop=True)
    
    # ROI mask dataframe
    roi_masks_df = dicom_df[dicom_df['SeriesDescription'] == 'ROI mask images']
    
    # Full mammogram dataframe
    full_mammograms_df = dicom_df[dicom_df['SeriesDescription'] == 'full mammogram images']

    cropped_df = dicom_df[dicom_df['SeriesDescription'] == 'cropped images']
    cropped_df = pd.merge(cropped_df,
                          mass_df,
                          left_on='PatientID',
                          right_on='uid')
    
    return dicom_df, full_mammograms_df, roi_masks_df, cropped_df

def pair_images(roi_masks_df, full_mammograms_df):
    # Extract the base patient ID from each dataframe
    roi_masks_df = roi_masks_df.copy()
    roi_masks_df['base_id'] = roi_masks_df['PatientID'].str.extract(r'(.*_[A-Z]+_[A-Z]+)')
    
    full_mammograms_df = full_mammograms_df.copy()
    full_mammograms_df['base_id'] = full_mammograms_df['PatientID']
    
    # Merge the two dataframes on the base_id column
    merged_df = pd.merge(
        full_mammograms_df,
        roi_masks_df,
        on='base_id',
        suffixes=('_full', '_roi')
    )
    
    print(f"Successfully paired {len(merged_df)} mammogram/ROI pairs.")
    
    initial_len = len(merged_df)
    # Remove entries where the dimensions of the full image and ROI don't match
    merged_df = merged_df[
        (merged_df['Rows_full'] == merged_df['Rows_roi']) &
        (merged_df['Columns_full'] == merged_df['Columns_roi'])
        ]
    print(f"Removed {initial_len - len(merged_df)} entries due to mismatched dimensions.")
    
    # Group ROIs of the same image together
    paired_df = merged_df.groupby('image_path_full').agg({
        'image_path_roi': list,
        'PatientID_full': 'first',
        'base_id': 'first'
    }).reset_index()
    
    print(f"Consolidated into {len(paired_df)} image entries.")
    
    return paired_df, merged_df

def get_model_file_path(prompt: str, initial_directory: str, loop: bool = True) -> str | None:
    """
    Retrieve the file path of a selected model file using a graphical file dialog. Provides a simple prompt for
    the user and allows for looping interaction until a file is selected or the process is exited.

    :param prompt: Message to display to the user in the input prompt.
    :type prompt: str
    :param initial_directory: Initial directory to open in the file dialog.
    :type initial_directory: str
    :param loop: If set to True, the function will loop until the user provides valid input.
        Defaults to True.
    :type loop: bool
    :return: The file path of the selected model file, an empty string if 'n' is chosen,
        or None if the loop is ended without selection.
    :rtype: str | None
    """

    run_once = False
    while loop and not run_once:
        run_once = True

        check = input(prompt)

        if check == 'y':
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            return filedialog.askopenfilename(parent=root,
                                              initialdir=initial_directory,
                                              title="Select a file",
                                              filetypes=(("Model Files", "*.pth"), ("All Files", "*.*"))
                                              )
        elif check == 'n':
            return ''
    return None

if __name__ == "__main__":
    clear_checkpoints()