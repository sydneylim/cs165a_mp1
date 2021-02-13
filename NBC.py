import numpy as np
import pandas as pd

df = pd.read_csv("covid_train.csv")

Dict = {}

df.loc[df['date_died'] == '9999-99-99', 'date_died'] = 0
df.loc[df['date_died'] != '9999-99-99', 'date_died'] = 1

for col in ['sex', 'patient_type', 'intubed', 'pneumonia', \
            'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
            'hypertension', 'other_disease', 'cardiovascular', \
            'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
            'covid_res', 'icu']:

    df.loc[np.logical_and(df[col] != 1, df[col] != 2), col] = 0


for col in df:
    if col not in ['entry_date', 'date_symptoms', 'date_died', 'age']:
        for val in df[col].unique():
            Dict[(col, val)] = sum(df[col] == val)

for key in Dict:
    print(key,':', Dict[key])


