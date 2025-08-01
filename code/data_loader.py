
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# dataloader for 2023
def read_dataset2023(data_name):
    data_dir = './dataset/'
    data = pd.read_csv(data_dir + 'data2023.csv')
    # Encode the last column (labels)
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
    # Remove rows containing inf or nan values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    # Perform random oversampling for classes with fewer than 600 samples
    class_counts = data.iloc[:, -1].value_counts()
    classes_to_oversample = class_counts[class_counts < 600].index
    oversampled_data = data.copy()

    for class_label in classes_to_oversample:
        class_data = data[data.iloc[:, -1] == class_label]
        samples_needed = 600 - len(class_data)
        # Random sampling with replacement to get the required number of samples
        oversampled_rows = class_data.sample(n=samples_needed, replace=True)
        oversampled_data = pd.concat([oversampled_data, oversampled_rows])

    data = oversampled_data
    print("Class distribution after oversampling:")
    print(data.iloc[:, -1].value_counts())
    # Print column labels (headers)
    # print("Original class distribution:")
    # print(data.iloc[:, -1].value_counts())
    return data.values[:,:-1], data.values[:,-1]

# dataloader for 2017
def read_dataset2017(data_name):
    data_dir = './dataset/'
    data = pd.read_csv(data_dir + 'data2017.csv')
    data = data.sample(frac=0.005, random_state=42).reset_index(drop=True)
    print(f"Original data size: {len(data)}, After 10% sampling: {len(data)}")# Encode the last column (labels)
    # Encode the last column (labels)
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
    # Remove rows containing inf or nan values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    # Perform random oversampling for classes with fewer than 600 samples
    class_counts = data.iloc[:, -1].value_counts()
    classes_to_oversample = class_counts[class_counts < 600].index
    oversampled_data = data.copy()

    for class_label in classes_to_oversample:
        class_data = data[data.iloc[:, -1] == class_label]
        samples_needed = 600 - len(class_data)
        # Random sampling with replacement to get the required number of samples
        oversampled_rows = class_data.sample(n=samples_needed, replace=True)
        oversampled_data = pd.concat([oversampled_data, oversampled_rows])

    data = oversampled_data
    print("Class distribution after oversampling:")
    print(data.iloc[:, -1].value_counts())
    return data.values[:,:-1], data.values[:,-1]

# dataloader for 2024
def read_dataset2024(data_name):
    data_dir = './dataset/'
    data = pd.read_csv(data_dir + 'Train_sample2024.csv')
    # data = data.replace({'Recon-Port_Scan': 'Recon', 'Recon-OS_Scan': 'Recon', 'DDoS': 'DoS'}, regex=True)
    # data = data.replace({'TCP_IP-DoS': 'TCP_IP_DoS'}, regex=True)
    data = data.replace({'Recon-Port_Scan': 'Recon', 'Recon-OS_Scan': 'Recon', 'Recon-VulScan': 'Recon', 'Recon-Ping_Sweep': 'Recon', 'DDoS': 'DoS'}, regex=True)
    # Encode the last column (labels)
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
    # Print mapping between numeric labels and original labels
    print("Label mapping:")
    for i, class_name in enumerate(le.classes_):
        print(f"{i}: {class_name}")
    # Remove rows containing inf or nan values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    # Perform random oversampling for classes with fewer than 600 samples
    class_counts = data.iloc[:, -1].value_counts()
    classes_to_oversample = class_counts[class_counts < 600].index
    oversampled_data = data.copy()

    for class_label in classes_to_oversample:
        class_data = data[data.iloc[:, -1] == class_label]
        samples_needed = 600 - len(class_data)
        # Random sampling with replacement to get the required number of samples
        oversampled_rows = class_data.sample(n=samples_needed, replace=True)
        oversampled_data = pd.concat([oversampled_data, oversampled_rows])

    data = oversampled_data
    print("Class distribution after oversampling:")
    print(data.iloc[:, -1].value_counts())
    return data.values[:,:-1], data.values[:,-1]

# dataloader for demo
# def read_dataset(data_name):
#     '''
#     =====A demo dataset======
#     '''
#     if data_name == 'demo':
#         data_dir = './dataset/demo/'
#         class_name = ['normal.', 'neptune.', 'smurf.', 'back.']
#         if not os.path.exists(data_dir + 'pro_data.csv'):
#             p = pd.read_csv(data_dir + 'kddcup.data_10_percent', error_bad_lines=False, header=None)
#             p_part = pd.DataFrame()
#             for i1 in class_name:
#                 p_part = pd.concat([p_part, p[p[41]==i1]], axis = 0)
#             for i2 in [1, 2, 3]:
#                 key_ls = list((p_part[i2].value_counts()).keys())
#                 for i3 in key_ls:
#                     p_part[i2][p_part[i2]==i3] = key_ls.index(i3)
#             for i4 in class_name:
#                 p_part[41][p_part[41]==i4] = class_name.index(i4)
#             p_part.to_csv(data_dir + 'pro_data.csv', index=False)
#         data = pd.read_csv(data_dir + 'pro_data.csv')
#         print("Original class distribution:")
#         print(data.iloc[:, -1].value_counts())
#         return data.values[:,:-1], data.values[:,-1]
