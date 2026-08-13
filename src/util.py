import os
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import kagglehub
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import torch
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from matplotlib import pyplot as plt


def clear_checkpoints():
    path = "../models/Resnet/checkpoints/"
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

def _grouped_split(dataframe: pd.DataFrame,
                   test_size: float,
                   random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve one stratified, patient-grouped fold off a dataframe that already has a 'patient' column.

    :param dataframe: Dataframe carrying 'patient' and 'pathology' columns.
    :type dataframe: pandas.DataFrame
    :param test_size: Approximate fraction of rows to place in the second returned split.
    :type test_size: float
    :param random_state: Seed controlling the shuffling of patient groups.
    :type random_state: int
    :return: A (larger_split, smaller_split) tuple.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """

    # n_splits controls the held-out fraction: 0.2 -> 5 folds, and one fold is held out.
    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    keep_idx, held_idx = next(splitter.split(dataframe,
                                             dataframe['pathology'],
                                             dataframe['patient']))

    return dataframe.iloc[keep_idx], dataframe.iloc[held_idx]

def patient_grouped_split(dataframe: pd.DataFrame,
                          val_size: float | None = None,
                          test_size: float = 0.2,
                          random_state: int = 42,
                          verbose: bool = True) -> tuple[pd.DataFrame, ...]:
    """
    Split a dataframe by patient so that no patient's images appear in more than one split.

    A single patient contributes several cropped images to the CBIS-DDSM dataset (left/right
    breast, CC/MLO views, and one crop per abnormality). Splitting on rows therefore leaks the
    same lesion across splits and inflates the reported scores. This function groups every row by
    its ``P_#####`` patient number so that all crops belonging to one patient land in the same
    split, while still keeping the benign/malignant ratio balanced.

    Pass ``val_size`` to get a three-way train/validation/test split. The validation set is what
    the scheduler and model selection should look at; the test set should be touched only once,
    for the final reported numbers.

    :param dataframe: Dataframe to split. Must contain a 'PatientID' and a 'pathology' column.
    :type dataframe: pandas.DataFrame
    :param val_size: Approximate fraction of the *whole* dataframe to place in a validation set.
        If None, no validation set is produced and only two splits are returned. Defaults to None.
    :type val_size: float | None
    :param test_size: Approximate fraction of rows to place in the test set. Because whole
        patients are kept together the realised fraction will not match exactly. Defaults to 0.2.
    :type test_size: float
    :param random_state: Seed controlling the shuffling of patient groups. Defaults to 42.
    :type random_state: int
    :param verbose: If True, print the resulting split sizes and assert that no patient is shared
        between any two splits. Defaults to True.
    :type verbose: bool
    :return: A (train_df, test_df) tuple, or (train_df, val_df, test_df) when val_size is given.
        The added 'patient' column is retained on each split.
    :rtype: tuple[pandas.DataFrame, ...]
    """

    dataframe = dataframe.copy()

    # Group on the bare patient number so that the Mass-Training / Mass-Test prefix, the
    # left/right side, the CC/MLO view and the abnormality index are all ignored.
    dataframe['patient'] = dataframe['PatientID'].str.extract(r'(P_\d+)')

    missing = dataframe['patient'].isna().sum()
    if missing:
        raise ValueError(f"Could not extract a patient number from {missing} PatientID value(s).")

    # Hold out the test set first so that its size does not depend on whether a validation
    # set was requested, keeping test results comparable across both modes.
    remainder_df, test_df = _grouped_split(dataframe, test_size, random_state)

    if val_size is None:
        splits = [('Train', remainder_df), ('Test', test_df)]
    else:
        # val_size is expressed as a fraction of the whole dataframe, but it is carved out of
        # the remainder, so rescale it against what is actually left after the test split.
        val_fraction = val_size / (1 - test_size)
        train_df, val_df = _grouped_split(remainder_df, val_fraction, random_state)
        splits = [('Train', train_df), ('Val', val_df), ('Test', test_df)]

    if verbose:
        total = len(dataframe)
        for name, split in splits:
            print(f"{name:<5} {len(split):>5} rows ({len(split) / total:>5.1%}), "
                  f"{split['patient'].nunique():>4} patients, "
                  f"malignant fraction {(split['pathology'] == 'MALIGNANT').mean():.3f}")

        for (name_a, split_a), (name_b, split_b) in combinations(splits, 2):
            overlap = set(split_a['patient']) & set(split_b['patient'])
            assert not overlap, (f"Patient leakage: {len(overlap)} patient(s) shared between "
                                 f"{name_a} and {name_b}.")
        print(f"No patient is shared between any of the {len(splits)} splits.")

    return tuple(split for _, split in splits)

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

def run_inference(model, loader, criterion, device) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run one no-grad pass over a loader and return the labels, malignant probabilities and loss.

    Probabilities are returned rather than hard predictions so that callers can compute
    threshold-independent metrics such as ROC-AUC, or apply an operating threshold other
    than the implicit 0.5 that argmax would use.

    :param model: Model to evaluate.
    :param loader: DataLoader to iterate over.
    :param criterion: Loss function, used to accumulate the average loss over the split.
    :param device: Device to run inference on.
    :return: A (labels, malignant_probabilities, mean_loss) tuple. The two arrays are aligned
        and both have one entry per sample in the loader's dataset.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float]
    """

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)

            # Column 1 is the malignant logit, so this is P(malignant) per sample.
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (np.asarray(all_labels),
            np.asarray(all_probs),
            running_loss / len(loader.dataset))

def evaluate_model(model, loader, criterion, device, split_name: str = 'Validation',
                   threshold: float = 0.5) -> dict:
    """
    Evaluate a model on a split and report accuracy, a classification report, ROC-AUC,
    a confusion matrix and an ROC curve.

    ROC-AUC is reported alongside accuracy because it is threshold-independent: it measures how
    well the model ranks malignant cases above benign ones, regardless of where the decision
    boundary sits. That makes it unaffected by the class weighting in the loss, which shifts the
    operating point but not the underlying ranking.

    :param model: Model to evaluate.
    :param loader: DataLoader for the split being evaluated.
    :param criterion: Loss function used to report the split's loss.
    :param device: Device to run inference on.
    :param split_name: Name used to label printed metrics and plot titles. Defaults to 'Validation'.
    :type split_name: str
    :param threshold: Probability above which a case is called malignant. Defaults to 0.5,
        which matches argmax. Lower it to trade false positives for fewer false negatives.
    :type threshold: float
    :return: A dict of the computed metrics, for collecting results across runs.
    :rtype: dict
    """

    labels, probs, loss = run_inference(model, loader, criterion, device)
    preds = (probs >= threshold).astype(int)

    accuracy = (preds == labels).mean()
    auc = roc_auc_score(labels, probs)

    print(f'{split_name} Loss: {loss:.4f}')
    print(f'{split_name} Accuracy: {accuracy:.4f}')
    print(f'{split_name} ROC-AUC: {auc:.4f}')
    if threshold != 0.5:
        print(f'(decision threshold: {threshold:.3f})')

    print('\nClassification Report:')
    print(classification_report(labels, preds, target_names=['benign', 'malignant']))

    cm = confusion_matrix(labels, preds)

    # Row 1 of the matrix is the malignant class, so [1, 0] is a malignant case predicted
    # benign: the false negative that matters most here.
    false_negatives = cm[1, 0]
    sensitivity = cm[1, 1] / cm[1].sum()
    print(f'False negatives (malignant predicted benign): {false_negatives} of {cm[1].sum()} '
          f'-> sensitivity {sensitivity:.3f}')

    fpr, tpr, _ = roc_curve(labels, probs)

    fig, (cm_ax, roc_ax) = plt.subplots(1, 2, figsize=(13, 5.5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benign', 'malignant'])
    disp.plot(cmap=plt.cm.Blues, ax=cm_ax, colorbar=False)
    cm_ax.set_title(f'{split_name} Confusion Matrix')

    roc_ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc:.3f})')
    roc_ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance (AUC = 0.500)')

    # Mark where the current threshold actually sits on the curve.
    roc_ax.plot(1 - cm[0, 0] / cm[0].sum(), sensitivity, 'o', color='red', markersize=8,
                label=f'Threshold {threshold:.2f}')

    roc_ax.set_xlabel('False Positive Rate (1 - specificity)')
    roc_ax.set_ylabel('True Positive Rate (sensitivity)')
    roc_ax.set_title(f'{split_name} ROC Curve')
    roc_ax.legend(loc='lower right')
    roc_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {'split': split_name, 'loss': loss, 'accuracy': accuracy, 'auc': auc,
            'sensitivity': sensitivity, 'false_negatives': int(false_negatives),
            'threshold': threshold}

if __name__ == "__main__":
    clear_checkpoints()
