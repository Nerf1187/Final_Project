---
title: Deep Learning Breast Cancer Detection
subtitle: Using Computer-Aided Detection (CAD) to Assist Radiologists
date: 06.29.2026
author: Anthony Youssef
csl: ./ieee.csl
bibliography: ./references.bib
---

# 1 Introduction

## 1.1 Background: Breast Cancer and Detection Techniques

Breast cancer remains one of the most prevalent and globally impactful non-communicable diseases among females. According to the World Health Organization (WHO), there were an estimated 2.3 million new women diagnosed with breast cancer and 670,000 deaths globally in 2022 alone. The American Cancer Society further estimates that approximately 13% of women in the United States will develop breast cancer at some point in their lifetime[@acs_stats;@who]. Various risk factors contribute to its development, including age, family history, genetic mutations (such as BRCA1/BRCA2), and environmental factors.

Because breast cancer outcomes are heavily dependent on the stage at diagnosis, early and accurate detection combined with prompt treatment is paramount. When detected early and managed in well-resourced healthcare systems, the 5-year survival rate exceeds 90% [@acs_sr;@nci_breasts]. However, in low-to-middle-income countries where early screening is less accessible, mortality rates are disproportionately higher. This disparity highlights the critical global need for highly accurate, scalable, and accessible screening tools.

Currently, X-ray mammography is considered the "gold standard" for frontline breast cancer screening. It is a widely available, non-invasive imaging modality that uses low-dose X-rays to detect structural abnormalities [@wang_FO]. Radiologists typically classify these findings using the standardized BI-RADS (Breast Imaging Reporting and Data System) framework, looking for specific anomalies such as microcalcifications, architectural distortions, and masses (differentiating between benign, well-circumscribed masses and potentially malignant, spiculated ones).

Despite its widespread use, traditional mammography has notable clinical limitations. Most prominently, it exhibits reduced sensitivity (dropping to approximately 85% or lower) in women with dense breast tissue, as the dense glandular tissue appears white in the scan, the exact same color as potentially malignant tumors, thereby visually obscuring them. An example of this can be seen in @fig-mammograms. While alternative imaging techniques exist, such as Ultrasound (which avoids ionizing radiation and handles dense tissue better but is highly operator-dependent) and Magnetic Resonance Imaging (MRI) (which offers up to 95% sensitivity but is prohibitively expensive and largely inaccessible for routine screening), mammography remains the primary global screening tool [@wang_FO].

![Comparison of non-dense and dense breast tissue (Image from @mam_ex)](./mammogram_example.png){#fig-mammograms}

## 1.2 Project Concept and Motivation

While mammography is indispensable, the manual interpretation of these medical images places a massive cognitive and visual load on radiologists. In a standard clinical setting, radiologists can perform up to 150+ screenings depending on the region, country, and hospital. This, combined with long (50+ hours per week), irregular work hours leads to severe fatigue and burnout [@ieong_worklod;@sunshine_work_hrs]. Because subtle malignancies can be incredibly difficult to differentiate from healthy dense tissue, this highly repetitive manual inspection is prone to human error and fatigue. Furthermore, it suffers from significant inter-observer variability, meaning different doctors might interpret the exact same scan differently depending on their experience level [@joshua_sd;@zhang_yolo].

Studies have shown that mammographic detection accuracy improves by 5–15% when scans are independently evaluated by a second radiologist [@chelpin_springer;@brem_ajr]. However, "double-reading" is not standard clinical practice globally due to severe shortages in specialized medical manpower and strict time constraints.

To bridge this gap, the medical field introduced Computer-Aided Diagnosis (CAD) systems. However, early generations of CAD relied on traditional machine learning and hand-crafted or statistically inferred features [@joshua_sd;@zhang_yolo]. These legacy systems were notorious for producing exceptionally high false-positive rates, which led to "alert fatigue," where the software flagged so many benign anomalies that it actually slowed radiologists down rather than assisting them.

This project is motivated by the need to overcome these legacy limitations by leveraging recent advancements in modern Artificial Intelligence (AI). Deep learning algorithms, specifically Convolutional Neural Networks (CNNs), represent a paradigm shift because they do not rely on human-engineered features; instead, they automatically learn complex, hierarchical, quantitative features directly from pixel data. This system is not intended to replace clinicians, but rather to act as a tireless, reliable "second opinion." By rapidly triaging images and automatically flagging potential malignancies with high precision, the model aims to minimize false negatives, reduce diagnostic bottlenecks, and assist healthcare professionals in making faster, data-driven decisions without succumbing to alert fatigue.

## 1.3 Project Aims

The primary aim of this project is to develop, train, and evaluate a robust deep learning computer vision pipeline capable of analyzing mammograms to automatically localize and classify breast tumors. To achieve this, the project is broken down into the following specific technical objectives:

* **Data Engineering:** Source, clean, and preprocess a recognized medical imaging benchmark dataset, specifically the CBIS-DDSM (Curated Breast Imaging Subset of DDSM).
* **Localization (Model 1):** Implement and train an object detection architecture, specifically a Faster R-CNN (Region-based Convolutional Neural Network), to accurately draw bounding boxes around Regions of Interest (ROIs) within the full mammogram scan.
* **Classification (Model 2):** Develop a secondary classification model (such as ResNet) that takes the localized crop from Model 1 and classifies the abnormality as either benign or malignant.
* **Clinical Safety Optimization:** Establish a rigorous evaluation framework using metrics appropriate for the medical domain (such as Mean Average Precision and F1-Score). Crucially, due to the high-stakes nature of cancer diagnosis, the system architecture and loss functions will be optimized to handle potential class imbalances and prioritize high Recall (Sensitivity). In a clinical context, a false negative (sending a sick patient home) is far more detrimental than a false positive (recommending a follow-up biopsy).

## 1.4 Project Template

In accordance with the module requirements, this project follows the “CM3015 Machine Learning and Neural Networks - Deep Learning Breast Cancer Detection” template. The applied methodology demonstrates an end-to-end machine learning pipeline, from data wrangling to multi-model orchestration, fulfilling the core learning outcomes of the brief.

# 2 Literature Review

## 2.1 Introduction to the Literature Review

The application of computer science to medical imaging has undergone a paradigm shift over the last two decades. As the global incidence of breast cancer rises, researchers have continually sought ways to automate and optimize the screening process to assist radiologists. This literature review traces the evolution of these efforts, exploring the transition from traditional Computer-Aided Diagnosis (CAD) systems to modern Deep Learning architectures. Specifically, it examines the current state-of-the-art in Convolutional Neural Networks (CNNs), the role of Region-Based CNNs (like Faster R-CNN) in tumor localization, and the standardization of benchmarking via datasets such as the CBIS-DDSM. By analyzing recent academic advancements, this review identifies the gaps in current methodologies that this project aims to address.

## 2.2 The Evolution and Limitations of Traditional CAD Systems

Computer-Aided Diagnosis (CAD) systems were introduced to clinical workflows in the late 1990s and early 2000s to act as a "second pair of eyes" for radiologists. These traditional CAD models relied heavily on hand-crafted feature extraction. Engineers and medical professionals would mathematically define specific visual features, such as edge gradients, texture contrasts, and geometric shapes, which were then fed into traditional machine learning classifiers like Support Vector Machines (SVMs) or Random Forests [@liew_cancers;@joshua_sd].

While initially promising, longitudinal clinical studies revealed significant limitations in traditional CAD systems. The primary drawback was an unacceptably high false-positive rate. Because hand-crafted features struggle to capture the complex, highly variable nature of breast tissue (especially dense tissue), traditional CAD systems frequently flagged healthy glandular tissue as suspicious. A comprehensive retrospective study comparing traditional CAD to modern AI approaches demonstrated that traditional CAD systems generated an average of 3.24 false marks per non-cancerous exam, compared to just 0.91 for deep learning models [@bahl_2025]. This high false-positive rate led to "alert fatigue," where radiologists learned to ignore the CAD system entirely, or worse, ordered unnecessary and invasive biopsies that caused severe patient anxiety and financial strain [@joshua_sd;@saber_nature].

## 2.3 The Deep Learning Paradigm Shift in Mammography

To overcome the limitations of hand-crafted features, the medical imaging field pivoted toward Deep Learning (DL), specifically Convolutional Neural Networks (CNNs). Unlike traditional machine learning, CNNs automatically learn hierarchical feature representations directly from raw pixel data. Lower layers learn fundamental features like edges and curves, while deeper layers synthesize these into complex semantic representations of masses or microcalcifications.

Recent literature demonstrates that DL architectures significantly outperform traditional CAD in both sensitivity and specificity. For example, Shen et al. (2019) demonstrated that deep CNNs could achieve an accuracy matching, and in some cases exceeding, that of experienced radiologists when classifying mammograms. Furthermore, by utilizing Transfer Learning, where models pre-trained on massive generalized datasets like ImageNet are fine-tuned on medical images, researchers have mitigated the issue of data scarcity in the medical domain.

## 2.4 Localization Techniques: The Role of Region-Based CNNs (Faster R-CNN)

In clinical practice, merely classifying an entire mammogram as "malignant" or "benign" is insufficient; a CAD system must also provide spatial interpretability by localizing the exact position of the suspicious mass. Consequently, object detection frameworks have become a focal point of recent research.

The Faster R-CNN (Region-based Convolutional Neural Network) architecture has emerged as a highly effective tool for this task. Faster R-CNN utilizes a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, enabling nearly cost-free region proposals. Studies evaluating Faster R-CNN on mammograms have shown promising results. For instance, a recent study applying a modified Faster R-CNN to mammogram datasets achieved a Mean Average Precision (mAP) of over 90% in detecting bounding boxes around lesions [@zhang_ui]. The literature suggests that while single-stage detectors like YOLO are faster, two-stage detectors like Faster R-CNN often provide the rigorous precision required for high-stakes medical imaging, making it the ideal choice for the localization phase (Model 1) of this project's pipeline.

## 2.5 Classification via ResNet Architectures

Once a mass is localized, the next objective is classification. Deep residual networks, specifically ResNet (e.g., ResNet-50), are heavily favored in breast cancer literature due to their ability to mitigate the "vanishing gradient" problem through skip connections. This allows for the training of exceptionally deep networks capable of capturing the nuanced textures that differentiate a benign fibroadenoma from a malignant invasive ductal carcinoma.

In multiple recent studies evaluating various CNN backbones for breast cancer detection, ResNet-50 was identified as the optimal architecture, achieving an Area Under the Curve (AUC) of over 95% on benchmark datasets when coupled with different patch sampling techniques [@shen_nature;@rajendran]. This highlights the efficacy of combining a localization model to extract a "patch" or Region of Interest (ROI) and passing it to a ResNet backbone for final classification, the exact hierarchical approach adopted by this project.

## 2.6 Benchmark Datasets: The CBIS-DDSM Standard

A recurring theme in the literature is the vital importance of standardized datasets for training and evaluating DL models. Historically, the Digital Database for Screening Mammography (DDSM) was the primary resource. However, it suffered from outdated compression formats and imprecise region-of-interest annotations.

To resolve this, Lee et al. (2017) released the Curated Breast Imaging Subset of DDSM (CBIS-DDSM). This dataset includes updated ROI segmentations, bounding boxes, and verified pathological diagnoses (Benign, Benign without Callback, and Malignant) evaluated by trained mammographers. The literature overwhelmingly relies on CBIS-DDSM as a benchmark for proving model efficacy. By utilizing the CBIS-DDSM dataset, this project aligns itself with established academic standards, ensuring that the evaluation metrics generated are directly comparable to state-of-the-art research.

## 2.7 Comparative Analysis of Prior Work

### 2.7.1 The Efficacy and Clinical Challenges of Deep Learning in Mammography

In a recent comprehensive review, Wang (2024) explored the transformative potential of deep learning-assisted X-ray mammography for breast cancer screening. The study highlights that while traditional mammography is the standard, it struggles with sensitivity in patients with dense breast tissue. Wang (2024) demonstrates that the integration of deep learning algorithms significantly enhances diagnostic accuracy, allowing for the detection of subtle abnormalities that manual inspection might miss.

However, the paper also critically addresses the ongoing challenges of deploying these models in real-world clinical environments. These hurdles include the necessity for large, high-quality, labeled datasets to prevent model overfitting, as well as the overarching issue of model generalizability across diverse patient populations. Although the CBIS-DDSM dataset is exceptional for training breast cancer detection models, it is one of only a few high-quality datasets for such a task. This scarcity is mainly due to the difficulty of obtaining and properly labeling data.

Wang’s findings strongly support the architectural choices of this project, specifically the use of the curated CBIS-DDSM dataset to provide high-quality standardized training data, and the implementation of a rigorous evaluation framework designed to address the clinical generalizability concerns raised in the literature.

### 2.7.2 The Clinical Impact of Computer-Aided Detection on Radiologist Sensitivity

To understand the baseline necessity for automated localization and classification tools, it is crucial to examine how "second opinions" physically impact screening outcomes. Brem et al. (2003) conducted a foundational multi-institutional trial evaluating whether early Computer-Aided Detection (CAD) systems could successfully identify breast cancers that radiologists had previously missed. The study retrospectively analyzed 377 screening mammograms where cancer was present but was initially interpreted as normal or benign by clinicians.

The findings were striking: the calculated sensitivity of the radiologists without any computer assistance was only 75.4%. However, when the missed cases were evaluated using a CAD system, the estimated sensitivity of the radiologists rose to 91.4%, representing a 21.2% increase in overall radiologist sensitivity. Furthermore, the study noted that while having two independent radiologists double-read a mammogram improves detection by 5–15%, this manual process is highly resource-intensive. This validates the core premise of this project: an automated deep learning pipeline (like the proposed Faster R-CNN) does not need to be perfect on its own; by simply flagging missed anomalies as a tireless, automated "second reader," it has the potential to dramatically improve diagnostic accuracy without straining hospital manpower.

# 3 Design

## 3.1 Project Overview & Template Identification

The objective of this project is to develop an automated computer vision pipeline capable of detecting and classifying breast cancer tumors in screening mammograms. By utilizing deep learning algorithms, the system aims to act as a reliable "second opinion" to assist radiologists in clinical diagnostics.

Once again, in accordance with module guidelines, this project follows the “CM3015 Machine Learning and Neural Networks - Deep Learning Breast Cancer Detection” template.

## 3.2 Domain and User Analysis

### 3.2.1 The Domain

This project operates within the high-stakes domain of healthcare and medical imaging. Unlike general-purpose image classification (e.g., identifying animals or vehicles), medical machine learning requires rigorous standards of clinical safety, interpretability, and fault tolerance. In this domain, the cost of a false negative (failing to detect a malignant tumor) is potentially fatal, meaning the system must be heavily biased toward high sensitivity (recall). Furthermore, medical imaging data is inherently complex; mammograms feature low contrast, fuzzy tumor boundaries, and high class-imbalance (most scans are healthy), which dictates specific data handling and loss-function requirements.

### 3.2.2 Target Users and Personas

The primary end-users of this system are specialized healthcare professionals, specifically screening mammographers, radiologists, and oncology triage teams. The system is explicitly not designed to be an autonomous diagnostic tool that replaces doctors. Instead, the persona is a highly trained but severely time-constrained professional suffering from visual fatigue. The system is designed to alleviate their cognitive load by pre-processing scans, drawing bounding boxes around suspicious regions, and offering a probabilistic classification (Benign vs. Malignant) to expedite the manual review process.

## 3.3 Design Requirements and Justification

A key design requirement is addressing the historical limitations of Computer-Aided Diagnosis (CAD) systems, which traditionally suffered from excessively high false-positive rates. To justify its clinical utility, this system must not overwhelm the user with "alert fatigue."

Therefore, a single-model approach (passing the entire mammogram into one classifier) was rejected. Single models often struggle to isolate microscopic features (like microcalcifications) against the massive background of a high-resolution 4K mammogram. Instead, a hierarchical, two-stage "Crop and Classify" design is required. By forcing an object-detection model to explicitly isolate the Region of Interest (ROI) first, we dramatically reduce the background noise before the secondary classification model makes its final prediction.

## 3.4 Overall Structure and Architectural Pipeline

The project architecture is divided into a two-stage sequential pipeline:

1. **Stage 1:** Localization (Model 1): The system first ingests a scaled-down mammogram image. A Faster R-CNN (Region-based Convolutional Neural Network) is tasked with scanning the image to locate potential masses. It outputs a bounding box coordinate (x, y, width, height) and a confidence score. Faster R-CNN was chosen over single-stage detectors (like YOLO) because its Region Proposal Network (RPN) generally yields the higher precision required for ambiguous medical textures.
2. **Stage 2:** Classification (Model 2): The bounding box coordinates generated by Model 1 are used to programmatically crop the localized abnormality from the original scan. This much smaller, isolated patch is then fed into a deep Residual Network (ResNet-50). The ResNet model extracts deep semantic features to classify the crop into one of two clinical categories: Benign or Malignant.

## 3.5 Technologies Stack and Key Methods

To execute this architecture, the following technology stack was selected:

* **Python:** The core programming language, chosen for its absolute dominance and extensive library support in the machine learning ecosystem.
* **PyTorch & Torchvision:** The primary deep learning framework. Beyond PyTorch’s intuitive dynamic computational graph, it was explicitly chosen over TensorFlow due to its superior hardware compatibility in this project's development environment. Recent versions of TensorFlow have deprecated native GPU support for Windows, requiring complex WSL2 workarounds. PyTorch provides seamless, native CUDA acceleration on newer Windows builds, ensuring the local GPU can be fully utilized without environment overhead. Torchvision provides the pre-trained Faster R-CNN and ResNet backbones.
* **OpenCV (cv2):** Used extensively in the data preprocessing pipeline. Specifically, OpenCV’s contour detection algorithms are required to automatically translate the binary mask images provided in the dataset (stored as JPGs where the tumor is pure white against a pure black background) into strictly defined bounding-box coordinates for the Faster R-CNN training loop.
* **Pandas:** Utilized to parse and merge the complex clinical CSV metadata associated with the dataset, matching patient IDs to their corresponding mass images and pathology reports.
* **Kaggle API:** Used to programmatically download the pre-processed dataset directly into the local development environment.

## 3.6 Data Management Strategy and Hardware Feasibility

### 3.6.1 Data Management

A major feasibility challenge in medical deep learning is data storage and hardware limitations. The raw CBIS-DDSM dataset hosted on The Cancer Imaging Archive (TCIA) exceeds 160 GB and uses obsolete, highly cumbersome DICOM compression formats.

To optimize the training pipeline, a data engineering workaround was implemented. The project utilizes a derivative dataset hosted on [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset). This version converted the raw scans into standard JPG formats while extracting the vital DICOM metadata into separate, easily parsed CSV files. According to the dataset documentation, this conversion maintains the same image quality and resolution as the original DICOM files. Crucially, this format translation reduced the total dataset footprint to a highly manageable 6.3 GB.

### 3.6.2 Justification for Local Hardware Execution Over Cloud Infrastructure {#local_specs}

While cloud-based machine learning environments like Google Colab are frequently utilized for academic projects, this system architecture explicitly avoids cloud dependency in favor of local hardware execution. The end-to-end pipeline is trained and evaluated entirely on an Alienware 16X Aurora AC16251 equipped with an Intel Core Ultra 9 275HX processor, 32 GB RAM, and an NVIDIA RTX 5070 GPU. This strategic engineering decision was driven by several severe limitations inherent to Google Colab’s free tier, which pose a direct risk to the feasibility of training a complex Faster R-CNN architecture.

First, evaluating the hardware specifications reveals a significant performance advantage locally. Google Colab’s free tier typically provides aging and heavily shared NVIDIA T4 GPUs with constrained RAM allocations. In contrast, the RTX 50-series graphics architecture provides vastly superior CUDA core counts and memory bandwidth, which accelerates the tensor calculations required for high-resolution medical imaging. Furthermore, Colab's compute allocation is highly dynamic and varied; access to GPUs or Tensor Processing Units (TPUs) is never guaranteed. During peak network times, the free tier aggressively throttles resource availability, which introduces unpredictable delays into the development cycle.

Second, deep learning workflows, particularly the iterative hyperparameter tuning required to optimize a two-stage object detection and classification pipeline, are highly time-intensive. Google Colab enforces a strict 12-hour maximum runtime limit on free instances, additionally terminating sessions even sooner due to idle timeouts or sudden resource reallocation. Training a Faster R-CNN model across a large number of epochs (potentially on Colab's free CPU) can easily exceed these limits, resulting in the loss of training progress mid-execution. Local execution entirely circumvents this constraint, allowing for uninterrupted, multi-day training loops.

Finally, data persistence and Input/Output (I/O) speeds were critical deciding factors. Virtual cloud instances are ephemeral; once a Google Colab session terminates, the virtual machine is wiped. Retaining the 6.3 GB CBIS-DDSM dataset between sessions would require integrating and mounting a personal Google Drive account. Not only does this risk exceeding the storage limits of a standard cloud drive, but reading thousands of high-resolution images across a network drive during PyTorch's `__getitem__` training loop creates a massive I/O bottleneck. By storing the dataset permanently on local solid-state storage, the PyTorch data loaders can operate at maximum read speeds, drastically reducing the overall epoch training time.

## 3.7 Testing and Evaluation Framework

Because the pipeline consists of two distinct models, the evaluation framework is bifurcated:

* **Localization Evaluation:** Model 1 is evaluated using Mean Average Precision (mAP) across varying Intersection over Union (IoU) thresholds. In medical imaging, drawing pixel-perfect boxes around fuzzy tumors is subjectively difficult even for human radiologists. Therefore, an mAP at an IoU of 0.50 (mAP_50) is considered the primary metric for successful region proposals.
* **Classification Evaluation:** Model 2 will be evaluated using standard classification metrics, but with heavily weighted priorities. While Overall Accuracy and F1-Score are recorded, Recall (Sensitivity) is the absolute most critical metric. The loss functions and thresholding will be tuned to minimize false negatives, as missing a malignant tumor is a catastrophic clinical failure.

## 3.8 Project Work Plan and Timeline

To ensure the systematic execution of this two-stage hierarchical deep learning pipeline, a comprehensive work plan was developed. The project is structured into eight sequential—potentially overlapping—phases, tracking the progression from initial research to the final end-to-end system evaluation. Tables -@tbl-midterm_work and -@tbl-final_work show the estimated week-by-week timeline for the prototype and final products respectively. Note that weeks 11, 12, 21, and 22 are missing. These weeks are reserved for coursework from other modules.

**Phase 1: Literature Review and Research**

- **Objective:** Establish the clinical need and technical foundation for the proposed architecture.
- **Tasks:** Review existing academic literature regarding the limitations of traditional CAD systems, the impact of dense breast tissue on screening sensitivity, and state-of-the-art computer vision architectures (such as Region Proposal Networks and Convolutional Neural Networks) for medical imaging. This phase will continue throughout the entire project as more research may be needed at any time.

**Phase 2: Dataset Selection and Exploration**

- **Objective:** Secure, analyze, and map the primary data source.
- **Tasks:** Find a suitable, publicly available, and widely used dataset for the project. Conduct comprehensive data exploration to map clinical metadata to the mammography scans, analyze the class distribution (Malignant vs. Benign), and establish the feasibility of executing the training pipeline on local hardware vs. cloud-based environments.

**Phase 3: Data Cleaning and Preprocessing**

- **Objective:** Standardize the raw medical imagery for neural network ingestion.
- **Tasks:** Filter out microcalcification samples to isolate mass abnormalities and remove scans with corrupted or mismatched Region of Interest (ROI) dimensions. Implement the `albumentations` library to resize images, apply probabilistic spatial augmentations, and convert binary ROI masks into exact mathematical bounding box coordinates. Finally, partition the dataset into a strict 80–20 train-test split.

**Phase 4: Model 1 Design, Training, and Testing (Feature Prototype)**

- **Objective:** Develop and evaluate the tumor localization architecture.
- **Tasks:** Implement a Faster R-CNN model with a ResNet-50-FPN backbone using PyTorch. Train the network utilizing native CUDA acceleration to identify and draw bounding boxes around suspected masses. Evaluate the model's success using visual prediction overlays and Mean Average Precision (mAP) metrics, specifically prioritizing the mAP_50 threshold.

**Phase 5: Preliminary Report Compilation**

- **Objective:** Document the system design and prove the feasibility of the prototype.
- **Tasks:** Synthesize the literature review, architectural design, data exploration, and Model 1 test results into a formal academic preliminary report. Identify known architectural limitations and draft strict methodological solutions for the subsequent phases.

**Phase 6: Model 2 Design, Training, and Testing**

- **Objective:** Develop the secondary mass classification architecture.
- **Tasks:** Implement a ResNet-50 classifier designed to ingest the cropped anomalies generated by Model 1. Train the model using the pathology-verified ground-truth bounding boxes coupled with spatial "jitter" augmentations to simulate real-world inaccuracies. Apply a weighted loss function to ensure the model prioritizes clinical sensitivity (Recall).

**Phase 7: Full Pipeline Integration and Testing**

- **Objective:** Evaluate the end-to-end automated diagnostic system.
- **Tasks:** Integrate Model 1 and Model 2 into a single continuous inference script. Feed unseen mammograms into the pipeline to evaluate the automated cropping and classification process. Tune Model 1's confidence threshold to mitigate false positives and benchmark the overall system's accuracy against existing clinical standards.

**Phase 8: Final Report and Deliverables**

- **Objective:** Finalize all technical and academic documentation for final submission.
- **Tasks:** Compile the final testing metrics, clean and comment the final source code repository, and write the final comprehensive project report.

| Phase |      W1      |      W2      |      W3      |      W4      |      W5      |      W6      |      W7      |      W8      |      W9      |     W10      |
|:------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 1     | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| 2     |              | $\checkmark$ |              |              |              |              |              |              |              |              |
| 3     |              | $\checkmark$ | $\checkmark$ |              |              |              |              |              |              |              |
| 4     |              |              |              | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |              |              |              |
| 5     |              |              |              |              |              | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
: Prototype Timeline (Weeks 1–10) {#tbl-midterm_work}

| Phase |     W13      |     W14      |     W15      |     W16      |     W17      |     W18      |     W19      |     W20      |     W23      |     W24      |
|:------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 1     | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| 2     | $\checkmark$ |              |              |              |              |              |              |              |              |              |
| 4     |              |              |              |              | $\checkmark$ | $\checkmark$ |              |              |              |              |
| 6     | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |              |              |              |              |
| 7     |              |              |              |              |              |              | $\checkmark$ | $\checkmark$ |              |              |
| 8     |              |              |              |              | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
: Final Product Timeline (Weeks 13–22) {#tbl-final_work}

# 4 Feature Prototype

## 4.1 Introduction to the Prototype

As outlined in the system architecture, this project utilizes a two-stage hierarchical deep learning pipeline. Because the secondary classification model (Model 2) relies entirely on the successful region proposals of the localization model (Model 1), the localization pipeline is the most critical technical dependency. Therefore, this feature prototype focuses on the implementation, training, and evaluation of Model 1 (Faster R-CNN) and the subsequent Region of Interest (ROI) extraction pipeline. Proving the feasibility of this step ensures the remainder of the project can be executed successfully.

## 4.2 Data Cleaning and Preprocessing

Before feeding the mammograms into the Faster R-CNN, a rigorous data cleaning and preprocessing pipeline was established to ensure high-quality inputs and stable model training.

During the initial data filtering phase, all microcalcification samples were explicitly removed from the dataset. This architectural decision was made to allow the current prototype to focus exclusively on the localization of mass abnormalities; the calcification data will be re-integrated into the pipeline for the final product submission. Furthermore, a quality-control check was implemented to remove any samples where the binary ROI mask dimensions did not perfectly match the dimensions of the corresponding original mammogram. This step was critical to prevent severe spatial misalignment errors during the OpenCV bounding-box generation process. Once the data was cleaned, it was divided using a standard 80–20 train-test split, ensuring a robust and entirely unseen dataset for prototype evaluation.

Following the data cleaning, the mammograms required structural preprocessing to meet the input tensor requirements of the neural network. As demonstrated in @lst-transforms, the `albumentations` Python library was utilized to construct sophisticated, synchronized transformation pipelines for both the training and testing datasets.

```{#lst-transforms .python lst-cap="Albumentations transformation pipeline for training and testing datasets."}
rcnn_train_transform = A.Compose(
    [
	    A.LongestMaxSize(1_024),
        A.PadIfNeeded(1_024, 1_024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)
rcnn_test_transform = A.Compose(
    [
	    A.LongestMaxSize(1_024),
        A.PadIfNeeded(1_024, 1_024),
        A.ToFloat(max_value=255.0),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)
```

- **Training Transformations (`rcnn_train_transform`):** Medical images natively come in widely varying resolutions. To standardize them without distorting the anatomical shapes, the `LongestMaxSize` function is used to scale the longest edge of every image to exactly 1,024 pixels while preserving the original aspect ratio. The `PadIfNeeded` function then applies necessary black padding to force the image into a uniform 1,024x1,024 square. To artificially expand the dataset and improve the model's ability to generalize against overfitting, probabilistic data augmentations are applied. These include a 50% probability of horizontal and vertical flipping, alongside a 20% probability of random brightness and contrast adjustments. Finally, `ToFloat` normalizes the pixel values to a 0–1 scale, and `ToTensorV2` converts the matrix for PyTorch computation. Finally, the `BboxParams` argument dictates that whenever an image is resized, padded, or flipped, its corresponding ground-truth bounding box coordinates (using the `pascal_voc` format of `[xmin, ymin, xmax, ymax]`) are mathematically updated in perfect synchronization.
- **Testing Transformations (`rcnn_test_transform`):** The testing pipeline mirrors the structural resizing, padding, and normalization required to feed the 1,024x1,024 tensors into the model. However, it explicitly omits all probabilistic augmentations (flips and brightness shifts) to guarantee a strictly deterministic and reliable evaluation on the 20% test split.

## 4.3 Dataset Exploration

The first functional step of the prototype was establishing a robust, automated data pipeline. To bypass the storage constraints of the original 160 GB TCIA dataset, the Kaggle API was utilized within the Python environment. The kagglehub library was explicitly called to securely and programmatically download the pre-processed 6.3 GB derivative of the CBIS-DDSM dataset directly to the local machine.

Upon extracting the data to the local `data/cbis-ddsm` directory, a significant data exploration phase was conducted. The Pandas library was utilized to parse and merge the clinical CSV metadata files, mapping each full-resolution mammogram to its corresponding bounding-box ROI mask(s) and pathological ground-truth labels (Benign vs. Malignant).

During this exploration, the class distribution of the masses was analyzed (see @fig-classes). The dataset categorizes tumors into Malignant, Benign, and 'Benign without callback' (B_NO_CB). To understand the overarching clinical challenge, the two benign classes were aggregated into a single B_ALL category. 

The dataset was found to contain 912 benign masses (53.8%) and 784 malignant masses (46.2%). While real-world clinical screening data suffers from extreme class imbalance, this specific dataset derivative is surprisingly well-balanced (a 1.16:1 ratio). Because of this near 50/50 distribution, the classification model (Model 2) is not at risk of majority-class collapse. Therefore, aggressive synthetic oversampling techniques like SMOTE or GANs are unnecessary. However, to ensure the model does not develop a slight bias toward the benign class, standard data augmentation (flipping, rotation) and a mildly weighted Cross-Entropy loss function will be utilized during training.

![CBIS-DDSM class distribution for masses](./class_distribution.png){#fig-classes}

## 4.4 Model 1 Implementation: Faster R-CNN Localization

With the data structured, the localization model was implemented. The prototype utilizes a pre-trained Faster R-CNN architecture with a ResNet-50-FPN (Feature Pyramid Network) backbone, imported via PyTorch's `torchvision.models` module.

A critical engineering challenge during implementation was translating the provided ROI data into a format the model could consume. The Kaggle dataset stores the ground-truth ROIs as binary JPG masks (where the tumor is pure white and the background is pure black), as seen in @fig-ex_images. To train the Faster R-CNN, these visual masks had to be mathematically converted into strict bounding box coordinates `[xmin, ymin, xmax, ymax]`. To achieve this, OpenCV (`cv2`) contour detection algorithms were integrated into the custom `__getitem__` function, demonstrated in @lst-get_item. During the training loop, the code dynamically reads the binary mask, finds the extreme contours of the white pixels, and returns the bounding box tensor required to calculate the loss functions.

```{#lst-get_item .python lst-cap="ROI-bounding box conversion"}
boxes = []
for r_path in roi_paths:
	# Read the mask
	mask = cv2.imread(r_path)
	pos = np.where(mask > 0)

	# Convert to bounding box format
	if len(pos[0]) > 0:
		xmin, xmax = np.min(pos[1]), np.max(pos[1])
		ymin, ymax = np.min(pos[0]), np.max(pos[0])
		boxes.append([xmin, ymin, xmax, ymax])

labels = [1] * len(boxes) if boxes else []

# Transform the image and bounding box(es)
transformed = self.transform(image=img, bboxes=boxes, labels=labels)
img_tensor = transformed['image']

# Transform box and label tensors into the proper type
if len(transformed['bboxes']) > 0:
	boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
	labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
else:
	boxes = torch.empty((0, 4), dtype=torch.float32)
	labels = torch.empty((0,), dtype=torch.int64)
```

Using the machine described in Section [3.6.2](#local_specs) and taking advantage of PyTorch's mixed-precision training (`autocast`) resulted in a training time of just over 1 hour for 10 epochs ($\approx \frac{6.6 \text{ min}}{\text{epoch}}$).

::: {#fig-ex_images layout-ncol=2 fig-cap="Example images"}

![Original image](./og_img.jpg){#fig-og_img width=70%}

![ROI Mask](./roi_mask.jpg){#fig-roi_mask width=70%}

:::

## 4.5 Prototype Evaluation and Output Visualization

To evaluate the success of the prototype, both quantitative metrics and qualitative visual inspections were employed.

* **Qualitative Visual Evaluation:** A custom `test_and_visualize` function was written using `matplotlib.patches` to overlay the model's predictions directly onto unseen test mammograms. Ground-truth bounding boxes were plotted in red, while the model's predicted bounding boxes (filtered by a confidence threshold) were plotted in green. Visual inspection of the prototype output confirms that the model is successfully ignoring healthy background tissue and actively identifying suspicious mass structures, demonstrating a functional Region Proposal Network (RPN). @fig-pred_vis demonstrates this achievement.
* **Quantitative Evaluation (mAP):** In object detection, Mean Average Precision (mAP) is the standard evaluation metric. The prototype achieved an impressive mAP of 0.6392 at an Intersection over Union (IoU) threshold of 0.50 (mAP_50). While the strict mAP (evaluated across IoU thresholds of 0.50 to 0.95) was lower at 0.2731, this discrepancy is a recognized phenomenon in medical imaging. Tumor boundaries in mammograms are inherently fuzzy, ambiguous, and highly subjective—even expert radiologists rarely achieve a 90% bounding box overlap with one another. Therefore, achieving a strong mAP_50 score of nearly 64% proves that the model is highly effective at its primary objective: isolating the general region of the anomaly so that it can be cropped and passed to the next stage of the pipeline.

![Faster R-CNN bounding box prediction](./test_and_visualize_ex.png){#fig-pred_vis width=80%}

## 4.6 Future Work

The prototype successfully demonstrates the feasibility of the most technically complex aspect of the project: automated localization. The immediate next phase involves developing Model 2, the ResNet-50 classifier.

To ensure Model 2 learns the precise morphological differences between benign and malignant masses, it will be trained strictly on the pathology-verified ground-truth bounding boxes rather than Model 1's predicted region proposals. This isolates variables and prevents the classifier from learning from imperfect or misaligned crops. However, to ensure the classification model remains robust when deployed in the final end-to-end pipeline, where it must rely on Model 1’s inevitably imperfect predictions, a spatial "jitter" data augmentation strategy will be implemented. During training, random coordinate shifts will be applied to the ground-truth bounding boxes before cropping. This simulates the slight localization inaccuracies of the Faster R-CNN, forcing Model 2 to accurately classify masses even when they are not perfectly centered.

Future work will additionally focus on improving Model 1's performance with a target mAP_50 score of 0.8. This will involve additional image preprocessing, hyperparameter tuning, and longer testing cycles.

## 4.7 Limitations

A known limitation of this hierarchical architecture is the risk of cascading errors: if Model 1 generates a false-positive region proposal by cropping healthy tissue, Model 2 will be forced into an out-of-distribution prediction. To mitigate this without unnecessarily increasing Model 1's confidence threshold (which would negatively impact clinical recall), future work will empirically test two solutions. The first approach will introduce a third 'Background' class to Model 2's training set to explicitly teach it to identify false-positive crops. The second approach will involve tuning a strict confidence threshold for Model 1 outputs. Both methods will be benchmarked against the validation set, and the final architectural implementation will be selected based on which strategy maximizes overall sensitivity while minimizing alert fatigue.

{{< pagebreak >}}

# References