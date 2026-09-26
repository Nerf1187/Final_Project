import os
from itertools import combinations
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog

import cv2
from dotenv import load_dotenv
import kagglehub
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
import torch


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
    
    paired_df['PatientBaseID'] = paired_df['base_id'].str.extract(r'(P_[0-9]{5})')
    
    print(f"Consolidated into {len(paired_df)} image entries.")
    
    return paired_df, merged_df

def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """
    Linearly scale and contrast-stretch image pixel intensities to the full [0, 255] dynamic range.

    Handles degenerate/constant images safely to avoid divide-by-zero, returning a uint8 array
    compatible with OpenCV and Albumentations pipelines.

    :param img: Input image array (2D grayscale or 3D RGB/BGR, float or integer dtype).
    :type img: numpy.ndarray
    :return: Contrast-stretched image array with uint8 dtype in range [0, 255].
    :rtype: numpy.ndarray
    """
    img_f = img.astype(np.float32)
    i_min = float(np.min(img_f))
    i_max = float(np.max(img_f))
    if i_max > i_min:
        norm = (img_f - i_min) / (i_max - i_min) * 255.0
        return np.clip(norm, 0, 255).astype(np.uint8)
    return img.astype(np.uint8) if img.dtype != np.uint8 else img.copy()

def generate_background_patches(paired_df: pd.DataFrame,
                                output_dir: str = "../data/cbis-ddsm/background_crops",
                                min_size: int = 128,
                                max_size: int = 512,
                                aspect_ratio_range: tuple[float, float] = (0.6, 1.6),
                                samples_per_image: int = 1,
                                max_total_samples: int | None = 600,
                                min_tissue_ratio: float = 0.70,
                                tissue_threshold: int = 10,
                                max_attempts_per_image: int = 100,
                                random_state: int = 42,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Extract non-lesion breast tissue background patches of varying sizes and aspect ratios
    from paired full mammograms and save them to disk.

    Ensures patches have 0% IoU with any annotated lesion ROI bounding box and contain
    at least min_tissue_ratio non-black breast tissue. Capped to max_total_samples to maintain
    balanced representation relative to lesion classes.

    :param paired_df: Consolidated DataFrame containing full mammograms and ROI mask lists.
    :type paired_df: pandas.DataFrame
    :param output_dir: Directory where background crop JPEG files will be saved.
    :type output_dir: str
    :param min_size: Minimum width or height of extracted patches in pixels. Defaults to 128.
    :type min_size: int
    :param max_size: Maximum width or height of extracted patches in pixels. Defaults to 512.
    :type max_size: int
    :param aspect_ratio_range: (min_ar, max_ar) tuple defining bounding box aspect ratio (w / h).
        Defaults to (0.6, 1.6).
    :type aspect_ratio_range: tuple[float, float]
    :param samples_per_image: Maximum number of patches to sample per full mammogram. Defaults to 1.
    :type samples_per_image: int
    :param max_total_samples: Maximum total background patches to extract across all images.
        Defaults to 600 (to match lesion class counts ~500-700). Pass None for no cap.
    :type max_total_samples: int | None
    :param min_tissue_ratio: Minimum ratio of non-zero tissue pixels in candidate crop. Defaults to 0.70.
    :type min_tissue_ratio: float
    :param tissue_threshold: Intensity threshold to distinguish breast tissue from background. Defaults to 10.
    :type tissue_threshold: int
    :param max_attempts_per_image: Number of random candidate boxes to sample per image. Defaults to 100.
    :type max_attempts_per_image: int
    :param random_state: Seed for reproducible patch sampling. Defaults to 42.
    :type random_state: int
    :param verbose: If True, prints progress and summary statistics. Defaults to True.
    :type verbose: bool
    :return: DataFrame containing ['PatientID', 'image_path', 'pathology', 'base_id'] metadata.
    :rtype: pandas.DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    # Shuffle paired_df deterministically to distribute sampling evenly across patients and views
    shuffled_df = paired_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    records = []
    for _, row in shuffled_df.iterrows():
        if max_total_samples is not None and len(records) >= max_total_samples:
            break

        img_path = row['image_path_full']
        roi_paths = row['image_path_roi']
        base_id = str(row['base_id'])
        patient_id_full = str(row.get('PatientID_full', base_id))

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        gray = img[:, :, 0] if img.ndim == 3 else img

        # Extract bounding boxes from associated ROI masks
        if isinstance(roi_paths, str):
            roi_paths = [roi_paths]

        roi_boxes = []
        for r_path in roi_paths:
            if not os.path.exists(r_path):
                continue
            mask = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            rx, ry, rw, rh = cv2.boundingRect(mask)
            if rw > 0 and rh > 0:
                roi_boxes.append((rx, ry, rx + rw, ry + rh))

        accepted_boxes = []
        for sample_idx in range(samples_per_image):
            if max_total_samples is not None and len(records) >= max_total_samples:
                break

            for _ in range(max_attempts_per_image):
                s = rng.integers(min_size, max_size + 1)
                ar = rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
                w = int(np.clip(round(s * np.sqrt(ar)), min_size, min(max_size, W)))
                h = int(np.clip(round(s / np.sqrt(ar)), min_size, min(max_size, H)))

                if w >= W or h >= H:
                    continue

                xmin = int(rng.integers(0, W - w + 1))
                ymin = int(rng.integers(0, H - h + 1))
                xmax = xmin + w
                ymax = ymin + h

                # 1. Enforce strict 0% IoU with all ground-truth lesion ROI boxes
                overlap = False
                for rx1, ry1, rx2, ry2 in roi_boxes:
                    if max(0, min(xmax, rx2) - max(xmin, rx1)) > 0 and max(0, min(ymax, ry2) - max(ymin, ry1)) > 0:
                        overlap = True
                        break
                if overlap:
                    continue

                # 2. Enforce 0% overlap with previously sampled background boxes in the same image
                for bx1, by1, bx2, by2 in accepted_boxes:
                    if max(0, min(xmax, bx2) - max(xmin, bx1)) > 0 and max(0, min(ymax, by2) - max(ymin, by1)) > 0:
                        overlap = True
                        break
                if overlap:
                    continue

                # 3. Check tissue quality: at least min_tissue_ratio non-black breast tissue
                crop_gray = gray[ymin:ymax, xmin:xmax]
                tissue_ratio = np.count_nonzero(crop_gray > tissue_threshold) / crop_gray.size
                if tissue_ratio < min_tissue_ratio:
                    continue

                # Valid background patch candidate
                crop_img = img[ymin:ymax, xmin:xmax]
                crop_img = min_max_normalize(crop_img)
                filename = f"{base_id}_BG_{sample_idx}.jpg"
                out_path = os.path.join(output_dir, filename).replace('\\', '/')
                cv2.imwrite(out_path, crop_img)

                accepted_boxes.append((xmin, ymin, xmax, ymax))
                records.append({
                    'PatientID': f"{patient_id_full}_BG_{sample_idx}",
                    'image_path': out_path,
                    'pathology': 'BACKGROUND',
                    'base_id': base_id,
                })
                break

    bg_df = pd.DataFrame(records, columns=['PatientID', 'image_path', 'pathology', 'base_id'])
    csv_path = os.path.join(output_dir, "background_crops.csv").replace('\\', '/')
    bg_df.to_csv(csv_path, index=False)

    if verbose:
        print(f"Generated {len(bg_df)} background patches across {bg_df['base_id'].nunique()} mammograms.")
        print(f"Saved crops and metadata to '{output_dir}'.")

    return bg_df

def compute_class_weights(dataframe: pd.DataFrame,
                          class_map: dict[str, int] | None = None,
                          device: torch.device | str | None = None) -> torch.Tensor:
    """
    Compute balanced class weights for CrossEntropyLoss based on inverse class frequencies.

    Formula: weight_c = total_samples / (num_classes * count_c)

    :param dataframe: DataFrame containing 'pathology' column.
    :type dataframe: pandas.DataFrame
    :param class_map: Mapping from pathology name to class index. Defaults to
        {'BACKGROUND': 0, 'BENIGN': 1, 'MALIGNANT': 2}.
    :type class_map: dict[str, int] | None
    :param device: Device to place the tensor on. Defaults to None.
    :type device: torch.device | str | None
    :return: 1D torch.Tensor of float32 class weights.
    :rtype: torch.Tensor
    """
    if class_map is None:
        class_map = {'BACKGROUND': 0, 'BENIGN': 1, 'MALIGNANT': 2}

    num_classes = len(class_map)
    counts = dataframe['pathology'].value_counts()
    total_samples = len(dataframe)

    weights = np.ones(num_classes, dtype=np.float32)
    for class_name, class_idx in class_map.items():
        count = counts.get(class_name, 0)
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
        else:
            weights[class_idx] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    tensor_weights = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor_weights = tensor_weights.to(device)
    return tensor_weights

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
            counts = split['pathology'].value_counts().to_dict()
            counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
            print(f"{name:<5} {len(split):>5} rows ({len(split) / total:>5.1%}), "
                  f"{split['patient'].nunique():>4} patients "
                  f"[{counts_str}]")

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
    while loop or not run_once:
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
    Run one no-grad pass over a loader and return the labels, probabilities and loss.

    Probabilities are returned rather than hard predictions so that callers can compute
    threshold-independent metrics such as multi-class ROC-AUC, or apply custom operating thresholds.

    :param model: Model to evaluate.
    :param loader: DataLoader to iterate over.
    :param criterion: Loss function, used to accumulate the average loss over the split.
        Can be None if loss computation is not needed.
    :param device: Device to run inference on.
    :return: A (labels, probabilities, mean_loss) tuple. labels is a 1D numpy array of shape (N,),
        probabilities is a numpy array of shape (N, num_classes), and mean_loss is the average loss.
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
            if criterion is not None:
                running_loss += criterion(outputs, labels).item() * images.size(0)

            # Softmax probabilities across class dimension: shape (B, num_classes)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_samples = len(loader.dataset) if hasattr(loader, 'dataset') and len(loader.dataset) > 0 else max(1, len(all_labels))
    mean_loss = running_loss / total_samples if criterion is not None else 0.0

    return (np.asarray(all_labels),
            np.asarray(all_probs),
            mean_loss)

def evaluate_model(model, loader, criterion, device, split_name: str = 'Validation',
                   class_names: list[str] | None = None) -> dict:
    """
    Evaluate a model on a split and report accuracy, a multi-class classification report,
    One-vs-Rest (OvR) macro ROC-AUC, a confusion matrix and One-vs-Rest ROC curves.

    :param model: Model to evaluate.
    :param loader: DataLoader for the split being evaluated.
    :param criterion: Loss function used to report the split's loss.
    :param device: Device to run inference on.
    :param split_name: Name used to label printed metrics and plot titles. Defaults to 'Validation'.
    :type split_name: str
    :param class_names: List of class names for display and reporting. Defaults to
        ['background', 'benign', 'malignant'].
    :type class_names: list[str] | None
    :return: A dict of the computed metrics.
    :rtype: dict
    """

    if class_names is None:
        class_names = ['background', 'benign', 'malignant']

    labels, probs, loss = run_inference(model, loader, criterion, device)

    # Determine predictions and number of classes
    if probs.ndim == 2 and probs.shape[1] > 1:
        num_classes = probs.shape[1]
        preds = np.argmax(probs, axis=1)
    elif probs.ndim == 1:
        num_classes = 2
        preds = (probs >= 0.5).astype(int)
    else:
        num_classes = len(class_names)
        preds = np.argmax(probs, axis=1)

    target_names = class_names[:num_classes] if len(class_names) >= num_classes else [f'class_{i}' for i in range(num_classes)]

    accuracy = float((preds == labels).mean()) if len(labels) > 0 else 0.0

    # Multi-class OvR Macro ROC-AUC calculation
    try:
        if probs.ndim == 2 and probs.shape[1] > 2:
            auc = float(roc_auc_score(labels, probs, multi_class='ovr', average='macro',
                                       labels=list(range(num_classes))))
        elif probs.ndim == 2 and probs.shape[1] == 2:
            auc = float(roc_auc_score(labels, probs[:, 1]))
        else:
            auc = float(roc_auc_score(labels, probs))
    except Exception:
        try:
            auc = float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
        except Exception:
            auc = float('nan')

    print(f'{split_name} Loss: {loss:.4f}')
    print(f'{split_name} Accuracy: {accuracy:.4f}')
    print(f'{split_name} Macro ROC-AUC: {auc:.4f}')

    print('\nClassification Report:')
    print(classification_report(labels, preds, target_names=target_names, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(len(target_names))))

    for c, c_name in enumerate(target_names):
        total_c = cm[c].sum()
        correct_c = cm[c, c]
        sens_c = correct_c / total_c if total_c > 0 else 0.0
        print(f"Sensitivity (Recall) for {c_name}: {correct_c}/{total_c} -> {sens_c:.3f}")

    fig, (cm_ax, roc_ax) = plt.subplots(1, 2, figsize=(13, 5.5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=cm_ax, colorbar=False)
    cm_ax.set_title(f'{split_name} Confusion Matrix')

    # Plot OvR ROC curves for each class
    if probs.ndim == 2 and probs.shape[1] > 1:
        for c, c_name in enumerate(target_names):
            y_c = (labels == c).astype(int)
            if len(np.unique(y_c)) > 1 and probs.shape[1] > c:
                fpr_c, tpr_c, _ = roc_curve(y_c, probs[:, c])
                try:
                    auc_c = roc_auc_score(y_c, probs[:, c])
                    roc_ax.plot(fpr_c, tpr_c, lw=2, label=f'{c_name} (AUC = {auc_c:.3f})')
                except Exception:
                    roc_ax.plot(fpr_c, tpr_c, lw=2, label=f'{c_name}')
    elif probs.ndim == 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc:.3f})')

    roc_ax.plot([0, 1], [0, 1], 'k--', lw=1, label=f'Chance / Macro OvR (AUC = {auc:.3f})')
    roc_ax.set_xlabel('False Positive Rate (1 - specificity)')
    roc_ax.set_ylabel('True Positive Rate (sensitivity)')
    roc_ax.set_title(f'{split_name} Multi-Class ROC Curves (OvR)')
    roc_ax.legend(loc='lower right')
    roc_ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'split': split_name,
        'loss': loss,
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(labels, preds, target_names=target_names,
                                                       output_dict=True, zero_division=0)
    }

if __name__ == "__main__":
    clear_checkpoints()
