import os
import re
import numpy as np
import pandas as pd

folder_path = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS/'
future_time_r = ['Q29', 'Q30', 'Q31']
upps = ['Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37']
impulsive = ['Q38', 'Q39', 'Q40']
for dirpath, dirs, files in os.walk(folder_path):
    if os.path.relpath(dirpath, folder_path).count(os.sep) == 2:
        for file in files:
            if file.endswith('.tsv') and 'surveyMedley' in file and not file.endswith('modified.tsv'):
                file_path = os.path.join(dirpath, file)
                print(f"Checking: {file_path}")
                modified_fname = file.replace("events", "events_modified")
                modified_path = os.path.join(dirpath, modified_fname)
                
                if os.path.exists(modified_path):
                    print(f"Already modified: {modified_path}")
                    continue
                
                print(f"Modifying file: {file_path}")
                name, ext = os.path.splitext(file)
                data = pd.read_csv(file_path, sep='\t')
    for index, row in data.iterrows():
        question = row['trial_type']
        coded_response = row['coded_response']
        if question in future_time_r:
            data.at[index, 'item_coding'] = "reverse"
            if not pd.isna(coded_response):
                if coded_response == 1.0:
                    data.at[index, 'coded_response'] = 1.0
                elif coded_response == 2.0:
                    data.at[index, 'coded_response'] = 0.75
                elif coded_response == 3.0:
                    data.at[index, 'coded_response'] = 0.5
                elif coded_response == 4.0:
                    data.at[index, 'coded_response'] = 0.25
                elif coded_response == 5.0:
                    data.at[index, 'coded_response'] = 0.0
                else:
                    data.at[index, 'coded_response'] = np.nan
        elif question in upps:
            data.at[index, 'item_coding'] = "reverse"
            if not pd.isna(coded_response):
                if coded_response == 1.0:
                    data.at[index, 'coded_response'] = 1.0
                elif coded_response == 2.0:
                    data.at[index, 'coded_response'] = 2/3
                elif coded_response == 3.0:
                    data.at[index, 'coded_response'] = 1/3
                elif coded_response == 4.0:
                    data.at[index, 'coded_response'] = 0.0
                else:
                    data.at[index, 'coded_response'] = np.nan
        elif question in impulsive:
            data.at[index, 'item_coding'] = "reverse"
            if not pd.isna(coded_response):
                if coded_response == 1.0:
                    data.at[index, 'coded_response'] = 1.0
                elif coded_response == 2.0:
                    data.at[index, 'coded_response'] = 0.0
                else:
                    data.at[index, 'coded_response'] = np.nan
        # for all other questions, item coding does not need to be modified and there are 5 valid response options
        else:
            if not pd.isna(coded_response):
                if coded_response == 1.0:
                    data.at[index, 'coded_response'] = 0.0
                elif coded_response == 2.0:
                    data.at[index, 'coded_response'] = 0.25
                elif coded_response == 3.0:
                    data.at[index, 'coded_response'] = 0.5
                elif coded_response == 4.0:
                    data.at[index, 'coded_response'] = 0.75
                elif coded_response == 5.0:
                    data.at[index, 'coded_response'] = 1.0
                else:
                    data.at[index, 'coded_response'] = np.nan
                data.to_csv(modified_path, sep='\t', index=False)
                print("Saving to:", modified_path)


    

