7hr 7min training time for Model 1 (15 epochs)

0.5, 0.6, 0.5 (metrics4)

|   Decision Threshold (τ_cls) | Sensitivity (Recall)   | Specificity (TNR)   | Diagnostic Accuracy   | PPV (Precision)   | NPV   |   False Negatives (Missed) |   False Positives |   Missed Lesions |   Extra Detections |   FP Marks/Image | Clinical Objective            |
|-----------------------------:|:-----------------------|:--------------------|:----------------------|:------------------|:------|---------------------------:|------------------:|-----------------:|-------------------:|-----------------:|:------------------------------|
|                         0.35 | 81.0%                  | 56.6%               | 63.1%                 | 40.5%             | 89.1% |                         28 |               175 |              233 |                377 |             0.69 | High Sensitivity Screening    |
|                          0.4 | 78.9%                  | 59.8%               | 64.9%                 | 41.7%             | 88.6% |                         31 |               162 |              233 |                377 |             0.69 | High Sensitivity Screening    |
|                         0.45 | 75.5%                  | 62.3%               | 65.8%                 | 42.2%             | 87.5% |                         36 |               152 |              233 |                377 |             0.69 | Balanced Operating Point      |
|                          0.5 | 72.8%                  | 65.5%               | 67.5%                 | 43.5%             | 86.8% |                         40 |               139 |              233 |                377 |             0.69 | High Specificity Confirmation |
|                         0.55 | 69.4%                  | 68.2%               | 68.5%                 | 44.3%             | 85.9% |                         45 |               128 |              233 |                377 |             0.69 | High Specificity Confirmation |


0.30, 0.35, 0.45 (metrics3)

|   Decision Threshold (τ_cls) | Sensitivity (Recall)   | Specificity (TNR)   | Diagnostic Accuracy   | PPV (Precision)   | NPV   |   False Negatives (Missed) |   False Positives |   Missed Lesions |   Extra Detections |   FP Marks/Image | Clinical Objective            |
|-----------------------------:|:-----------------------|:--------------------|:----------------------|:------------------|:------|---------------------------:|------------------:|-----------------:|-------------------:|-----------------:|:------------------------------|
|                         0.35 | 86.4%                  | 42.2%               | 54.0%                 | 35.3%             | 89.5% |                         20 |               233 |              187 |                577 |             1.05 | High Sensitivity Screening    |
|                          0.4 | 83.0%                  | 45.9%               | 55.8%                 | 35.9%             | 88.1% |                         25 |               218 |              187 |                577 |             1.05 | High Sensitivity Screening    |
|                         0.45 | 79.6%                  | 49.1%               | 57.3%                 | 36.3%             | 86.8% |                         30 |               205 |              187 |                577 |             1.05 | Balanced Operating Point      |
|                          0.5 | 75.5%                  | 54.1%               | 59.8%                 | 37.5%             | 85.8% |                         36 |               185 |              187 |                577 |             1.05 | High Specificity Confirmation |
|                         0.55 | 71.4%                  | 59.3%               | 62.5%                 | 39.0%             | 85.1% |                         42 |               164 |              187 |                577 |             1.05 | High Specificity Confirmation |


0.45, 0.60, 0.40 (metrics2)

|   Decision Threshold (τ_cls) | Sensitivity (Recall)   | Specificity (TNR)   | Diagnostic Accuracy   | PPV (Precision)   | NPV   |   False Negatives (Missed) |   False Positives |   Missed Lesions |   Extra Detections |   FP Marks/Image | Clinical Objective            |
|-----------------------------:|:-----------------------|:--------------------|:----------------------|:------------------|:------|---------------------------:|------------------:|-----------------:|-------------------:|-----------------:|:------------------------------|
|                         0.35 | 84.4%                  | 51.1%               | 60.0%                 | 38.6%             | 90.0% |                         23 |               197 |              215 |                478 |             0.87 | High Sensitivity Screening    |
|                          0.4 | 81.6%                  | 55.1%               | 62.2%                 | 39.9%             | 89.2% |                         27 |               181 |              215 |                478 |             0.87 | High Sensitivity Screening    |
|                         0.45 | 79.6%                  | 57.8%               | 63.6%                 | 40.8%             | 88.6% |                         30 |               170 |              215 |                478 |             0.87 | Balanced Operating Point      |
|                          0.5 | 76.9%                  | 60.5%               | 64.9%                 | 41.5%             | 87.8% |                         34 |               159 |              215 |                478 |             0.87 | High Specificity Confirmation |
|                         0.55 | 72.1%                  | 64.0%               | 66.2%                 | 42.2%             | 86.3% |                         41 |               145 |              215 |                478 |             0.87 | High Specificity Confirmation |


0.42, 0.60, 0.50 (metrics1)

|   Decision Threshold (τ_cls) | Sensitivity (Recall)   | Specificity (TNR)   | Diagnostic Accuracy   | PPV (Precision)   | NPV   |   False Negatives (Missed) |   False Positives |   Missed Lesions |   Extra Detections |   FP Marks/Image | Clinical Objective            |
|-----------------------------:|:-----------------------|:--------------------|:----------------------|:------------------|:------|---------------------------:|------------------:|-----------------:|-------------------:|-----------------:|:------------------------------|
|                         0.35 | 84.4%                  | 49.1%               | 58.5%                 | 37.7%             | 89.6% |                         23 |               205 |              212 |                544 |             0.99 | High Sensitivity Screening    |
|                          0.4 | 81.6%                  | 52.9%               | 60.5%                 | 38.7%             | 88.8% |                         27 |               190 |              212 |                544 |             0.99 | High Sensitivity Screening    |
|                         0.45 | 80.3%                  | 55.8%               | 62.4%                 | 39.9%             | 88.6% |                         29 |               178 |              212 |                544 |             0.99 | Balanced Operating Point      |
|                          0.5 | 77.6%                  | 58.8%               | 63.8%                 | 40.7%             | 87.8% |                         33 |               166 |              212 |                544 |             0.99 | High Specificity Confirmation |
|                         0.55 | 72.8%                  | 62.8%               | 65.5%                 | 41.6%             | 86.3% |                         40 |               150 |              212 |                544 |             0.99 | High Specificity Confirmation |
