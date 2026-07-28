import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np

# Visualization imports
from matplotlib import pyplot as plt
from matplotlib import patches

class CBISDDSM_RCNN_Dataset(Dataset):
    """
    Dataset class for managing CBIS-DDSM dataset configurations for object detection tasks.

    This class acts as a PyTorch Dataset for the CBIS-DDSM dataset, which is a collection
    of mammogram images annotated with region of interest (ROI) masks. The dataset is
    designed for object detection models such as Faster R-CNN. Each sample consists of an
    image and the corresponding bounding boxes (derived from ROI masks) along with labels.

    The class supports image transformations using libraries such as Albumentations,
    and provides a method to visualize images along with the annotated bounding boxes.

    :ivar df: Pandas dataframe containing metadata about the dataset. Each row corresponds
        to one mammogram image, and contains file paths to the image and associated
        ROI masks.
    :type df: pandas.DataFrame
    :ivar transform: Transformation pipeline for preprocessing the images and their bounding
        boxes. If None, no transformations will be applied.
    :type transform: callable or None
    """

    def __init__(self, dataframe: pd.DataFrame, transform: A.Compose | None = None):
        """
        Initializes the object with a dataframe and an optional transformation.

        This constructor creates an instance of the class with the provided
        dataframe. An optional transformation function can also be passed
        to modify or process the dataframe during further usage.

        :param dataframe: A pandas DataFrame containing the data to be
            processed.
        :type dataframe: pandas.DataFrame
        :param transform: An optional callable that takes the dataframe
            as input and returns a transformed dataframe. Defaults to None.
        :type transform: callable or None, optional
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieve an image tensor and its corresponding target dictionary containing bounding
        boxes and labels for regions of interest (ROIs) within the image.

        This method loads a mammogram image and its associated ROI masks. It calculates bounding
        boxes around each ROI, applies transformations if specified, and constructs a target dictionary
        suitable for training object detection models.

        :param idx: Index of the image and corresponding ROIs in the dataset.
        :type idx: int
        :return: A tuple containing the transformed tensor representation of the image and a
            target dictionary with bounding boxes, labels, and the image ID.
        :rtype: tuple[torch.Tensor, dict[str, torch.Tensor]]
        """

        # Get the image and ROI paths for the specified index
        img_path = self.df.iloc[idx]['image_path_full']
        roi_paths = self.df.iloc[idx]['image_path_roi']

        # Load the mammogram image and convert to RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load each ROI mask for the mammogram image
        boxes = []
        for r_path in roi_paths:
            # Read the mask and ensure it is the same size as the mammogram image
            mask = cv2.imread(r_path)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            pos = np.where(mask > 0)

            # Convert to bounding box format
            if len(pos[0]) > 0:
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

        labels = [1] * len(boxes) if boxes else []

        # Transform the image and bounding box(es)
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img_tensor = transformed['image']

            # Transform box and label tensors into the proper type
            if len(transformed['bboxes']) > 0:
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
        else:  # Default transformation
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4),
                                                                                                   dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "img_id": torch.tensor([idx]),
        }

        return img_tensor, target

    def visualize_sample(self, idx: int, title: str | None = None) -> None:
        """
        Visualize a sample from the dataset with its bounding box annotations.

        This method retrieves the image tensor and corresponding target annotations for
        the specified sample index. It then displays the image along with bounding boxes
        for each abnormality present in the target annotations. Bounding boxes are rendered
        as red rectangles with an accompanying label above each box. The title of the plot
        can be customized or defaults to "Sample Image and Bounding Box(es)" if not specified.

        :param idx: The index of the sample to retrieve and visualize.
        :type idx: int
        :param title: Optional title for the plot. Defaults to None, using the default title.
        :type title: str | None
        :return: None
        """

        # Get the image tensor and target annotations for the specified sample
        img_tensor, target = self[idx]

        try:
            # Image has multiple channels
            img_np = img_tensor.permute(1, 2, 0).numpy()
        except:
            # Image has only one channel
            img_np = img_tensor.squeeze(0).numpy()

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_np, cmap='gray')

        # Extract bounding box(es) from the target annotations
        boxes = target["boxes"].numpy()
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )

            ax.add_patch(rect)

            plt.text(xmin, ymin - 10, 'Abnormality', color='r', fontsize=12, weight='bold')

        plt.axis('off')
        plt.title("Sample Image and Bounding Box(es)" if not title else title)
        plt.show()
