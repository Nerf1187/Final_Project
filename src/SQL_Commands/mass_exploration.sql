CREATE VIEW IF NOT EXISTS Train_masses AS
SELECT patient_id, "image file path"
FROM Mass_descriptions
WHERE "image file path" LIKE '%Train%';

CREATE VIEW IF NOT EXISTS Test_masses AS
SELECT patient_id, "image file path"
FROM Mass_descriptions
WHERE "image file path" LIKE '%Test%';


SELECT *
FROM Train_masses
         JOIN Test_masses ON Train_masses.patient_id = Test_masses.patient_id
LIMIT 10;

SELECT COUNT(*)
FROM Dicom_info
WHERE PatientID NOT LIKE '%Mass%'
  AND PatientID NOT LIKE '%Calc%'
LIMIT 10;

SELECT *
FROM Dicom_info
WHERE PatientID LIKE '%Mass%';

SELECT COUNT(*)
FROM Mass_descriptions
WHERE pathology = 'BENIGN';

SELECT COUNT(*)
FROM Mass_descriptions
WHERE pathology = 'MALIGNANT';

SELECT *
FROM Mass_descriptions
WHERE "image file path" LIKE '%824428%';

SELECT SUM(CASE WHEN pathology = 'MALIGNANT' THEN 1 ELSE 0 END)               AS 'Malignant',
       SUM(CASE WHEN pathology = 'BENIGN' THEN 1 ELSE 0 END)                  AS 'Benign',
       SUM(CASE WHEN pathology = 'BENIGN_WITHOUT_CALLBACK' THEN 1 ELSE 0 END) AS 'B w/o CB',
       SUM(CASE WHEN pathology LIKE 'BENIGN%' THEN 1 ELSE 0 END)              AS 'B (ALL)'
FROM Mass_descriptions;

SELECT *, COUNT(*)
FROM Mass_descriptions
GROUP BY "image file path"
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC;

SELECT *
FROM Dicom_info
WHERE PatientID NOT LIKE '%Mass%'
  AND PatientID NOT LIKE '%Calc%';

SELECT *
FROM Mass_descriptions
WHERE patient_id LIKE '%00562%';

SELECT *
FROM Mass_descriptions
WHERE patient_id IN (SELECT PatientID FROM Dicom_info WHERE PatientID NOT LIKE '%Mass%' AND PatientID NOT LIKE '%Calc%');