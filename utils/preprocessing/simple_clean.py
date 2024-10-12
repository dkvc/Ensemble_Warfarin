"""
List of cols
1. PharmGKB Subject ID - id
2. Gender - male or female
3. Race - Asian, Black or African American, White, Unknown
4. Ethnicity - Hispanic or Latino, not Hispanic or Latino, Unknown
5. Age - messed up data -- contains range as well as single values
6. Height (cm) - height in cm
7. Weight (kg) - weight in kg
8. Indication for Warfarin Treatment - 8 different indications (can be single or multiple)
9. Comorbidities4 - Multiple comorbidities
10. Diabeties - 0, 1, NA
11. Congestive Heart Failure and/or Cardiomyopathy - 0, 1, NA
12. Valve Replacement - 0, 1, NA
13. Medications - Multiple medications
14. Aspirin - 0, 1, NA
15. Acetaminophen or Paracetamol (Tylenol) - 0, 1, NA
16. Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day - 0, 1, NA
17. Simvastatin (Zocor) - 0, 1, NA
18. Atorvastatin (Lipitor) - 0, 1, NA
19. Rosuvastatin (Crestor) - 0, 1, NA
20. Fluvastatin (Lescol) - 0, 1, NA
21. Lovastatin (Mevacor) - 0, 1, NA
22. Pravastatin (Pravachol) - 0, 1, NA
23. Cerivastatin (Baycol) - 0, 1, NA
24. Amiodarone (Cordarone) - 0, 1, NA
25. Carbamazepine (Tegretol) - 0, 1, NA
26. Phenytoin (Dilantin) - 0, 1, NA
27. Rifampin or Rifampicin - 0, 1, NA
28. Sulfonamide Antibiotics - 0, 1, NA
29. Macrolide Antibiotics - 0, 1, NA
30. Anti-fungal Azoles - 0, 1, NA
31. Herbal Medications, Vitamins, Supplements - 0, 1, NA
32. Target INR - 8 different values
33. Estimated Target INR Range Based on Indication - contains range as well as single values
34. Subject Reached Stable Dose of Warfarin - 0, 1, NA
35. Therapeutic Dose of Warfarin - 0, 1, NA
36. INR on Reported Therapeutic Dose of Warfarin - 0, 1, NA
37. Current Smoker - 0, 1, NA
38. CYP2C9 genotypes - Multiple genotypes, NA
39. Genotyped QC Cyp2C9*2 - Multiple genotypes, Blank
40. Genotyped QC Cyp2C9*3 - Multiple genotypes, Blank
41. Combined QC CYP2C9 - Multiple genotypes, Blank
42. VKORC1 genotype: 1639 G>A (3673) - A/A, A/G, G/G, NA
43. VKORC1 QC genotype: -1639 G>A (3673) - A/A, A/G, G/G, Blank
44. VKORC1 genotype: 497T>G (5808) - T/T, T/G, G/G, NA
45. VKORC1 QC genotype: 497T>G (5808) - T/T, T/G, G/G, Blank
46. VKORC1 genotype: 1173 C>T(6484) - C/C, C/T, T/T, NA, Blank
47. VKORC1 QC genotype: 1173 C>T(6484) - C/C, C/T, T/T, Blank
48. VKORC1 genotype: 1542G>C (6853) - G/G, G/C, C/C, NA
49. VKORC1 QC genotype: 1542G>C (6853) - G/G, G/C, C/C, Blank
50. VKORC1 genotype: 3730 G>A (9041) - G/G, G/A, A/A, NA
51. VKORC1 QC genotype: 3730 G>A (9041) - G/G, G/A, A/A, Blank
52. VKORC1 genotype: 2255 C>T (7566) - C/C, C/T, T/T, NA
53. VKORC1 QC genotype: 2255 C>T (7566) - C/C, C/T, T/T, Blank
54. VKORC1 genotype: -4451 C>A (861) - C/C, C/A, A/A, NA
55. VKORC1 QC genotype: -4451 C>A (861) - C/C, C/A, A/A, Blank
56. CYP4F2 1347C>T - Multiple values, NA
57. VKORC1 -1639 consensus - A/A, A/G, G/G, NA
58. VKORC1 497 consensus - T/T, T/G, G/G, NA
59. VKORC1 1173 consensus - C/C, C/T, T/T, NA
60. VKORC1 1542 consensus - G/G, G/C, C/C, NA
61. VKORC1 3730 consensus - G/G, G/A, A/A, NA
62. VKORC1 2255 consensus - C/C, C/T, T/T, NA
63. VKORC1 -4451 consensus - C/C, C/A, A/A, NA
64. Empty
65. Empty
"""

import os
import pandas as pd

def fill_missing_cols(dataset):
    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 2255 consensus'] == 'C/C')), 'VKORC1 -1639 consensus'] = 'G/G'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 2255 consensus'] == 'T/T')), 'VKORC1 -1639 consensus'] = 'A/A'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 2255 consensus'] == 'C/T')), 'VKORC1 -1639 consensus'] = 'A/G'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & (dataset['VKORC1 1173 consensus'] == 'C/C')), 'VKORC1 -1639 consensus'] = 'G/G'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna())
            & (dataset['VKORC1 1173 consensus'] == 'T/T')), 'VKORC1 -1639 consensus'] = 'A/A'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna())
            & (dataset['VKORC1 1173 consensus'] == 'C/T')), 'VKORC1 -1639 consensus'] = 'A/G'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 1542 consensus'] == 'G/G')), 'VKORC1 -1639 consensus'] = 'G/G'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 1542 consensus'] == 'C/C')), 'VKORC1 -1639 consensus'] = 'A/A'

    dataset.loc[((dataset['VKORC1 -1639 consensus'].isna()) 
            & ((dataset['Race'] != 'Black or African American') | dataset['Race'] != 'Unknown') 
            & (dataset['VKORC1 1542 consensus'] == 'C/G')), 'VKORC1 -1639 consensus'] = 'A/G'
    
def add_age_group(dataset):
    """
    Map:
    10-19: 1
    20-29: 2
    ...
    70-80: 7
    80-90: 8
    90+: 9
    """
    dataset['Age'] = dataset['Age'].map({'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, '50 - 59': 5, '60 - 69': 6, '70 - 79': 7, '80 - 89': 8, '90 - 99': 9, '90+': 10})

def add_col_enzymeinducer(dataset):
    dataset['Enzyme_inducer'] = dataset.apply(lambda row: 1 if (row['Carbamazepine (Tegretol)'] == 1 or row['Phenytoin (Dilantin)'] == 1 or row['Rifampin or Rifampicin'] == 1) else 0, axis=1)

def drop_redundant_cols(dataset):
    columns_to_drop = ['PharmGKB Subject ID', 'Cyp2C9 genotypes', 'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3',
       'Combined QC CYP2C9',
       'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'Medications', 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
       'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
       'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
       'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
       'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)', 
       'Carbamazepine (Tegretol)',
       'Phenytoin (Dilantin)', 'Rifampin or Rifampicin',
       'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
       'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
       'Ethnicity', 'Indication for Warfarin Treatment', 'Comorbidities',
       'Gender', 'Valve Replacement', 'Congestive Heart Failure and/or Cardiomyopathy',
       'Diabetes', 'Target INR', 'Estimated Target INR Range Based on Indication',
       'Subject Reached Stable Dose of Warfarin', 'INR on Reported Therapeutic Dose of Warfarin',
       'Current Smoker', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65'
       ]
    dataset.drop(columns_to_drop, axis=1, inplace=True)

def drop_rows_with_na_dosage(dataset):
    dataset.dropna(subset='Therapeutic Dose of Warfarin', inplace=True, axis=0)

def encode_one_hot(dataset):
    return pd.get_dummies(dataset)

def clean(dataset):
    print(f"Total rows in original data: {dataset.shape[0]}")

    fill_missing_cols(dataset)
    add_age_group(dataset)
    add_col_enzymeinducer(dataset)
    drop_redundant_cols(dataset)
    drop_rows_with_na_dosage(dataset)
    dataset = encode_one_hot(dataset)

    print(f"Total rows after simple cleaning: {dataset.shape[0]}")
    return dataset
    
def save(dataset):
    dataset_dir_location = './data/modified'
    os.makedirs(dataset_dir_location, exist_ok=True)

    dataset_location = dataset_dir_location + '/simple_cleaned.csv'
    print("Saving modified dataset to", dataset_location)
    dataset.to_csv(dataset_location, index=False)

if __name__ == "__main__":
    dataset = pd.read_csv('./data/original/warfarin.csv')
    cleaned_dataset = clean(dataset)
    save(cleaned_dataset)