import pandas as pd

# Load datasets
biomarker_data = pd.read_pickle('data/processed/biomarkers.pkl.bz2', compression='bz2')
clinical_data = pd.read_pickle('data/processed/clinical.pkl.bz2', compression='bz2')
mri_data = pd.read_pickle('data/processed/mri.pkl.bz2', compression='bz2')
tom_data = pd.read_pickle('data/processed/tomography.pkl.bz2', compression='bz2')
xray_data = pd.read_pickle('data/processed/xray.pkl.bz2', compression='bz2')
progression_data = pd.read_csv('data/progression.csv.bz2')

# Join datasets, excluding progression_data
datasets = [clinical_data, biomarker_data, mri_data, tom_data, xray_data]
data = datasets[0]
for dataset in datasets[1:]:
    data = pd.merge(data, dataset, on="Subject ID", how="inner")

data.reset_index(inplace=True)

# Extract the last four digits from Subject ID and convert to integer to remove leading zeros
data['Subject ID Last 4 Digits'] = data['Subject ID'].apply(lambda x: int(str(x)[-4:]))

# Merge datasets based on the matching Subject ID last 4 digits and ID
merged_data = pd.merge(data, progression_data, left_on='Subject ID Last 4 Digits', right_on='ID', how='inner')

# Drop the extra columns
merged_data.drop(['Subject ID Last 4 Digits', 'ID'], axis=1, inplace=True)

# Save merged dataset
merged_data.to_pickle('data/final/merged_data.pkl.bz2', compression='bz2')
