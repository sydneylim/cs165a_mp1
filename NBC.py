import numpy as np
import pandas as pd
import sys
from datetime import datetime as dt
from scipy.stats import norm, skewnorm
import random
import time

def main():

    train = sys.argv[1]
    valid = sys.argv[2]

    df = (preprocess(valid))
    #test_df = df

    # weights = {'sex': 1.8, 'patient_type': 0.2, 'entry_date': 2.0, 'date_symptoms': 1.6, 'date_died': 0.2, 'intubed': 0.2, 'pneumonia': 0.8, 'age': 0.4, 'pregnancy': 2.0, 'diabetes': 0.4, 'copd': 0.2, 'asthma': 0.2, 'inmsupr': 0.4, 'hypertension': 0.6000000000000001, 'other_disease': 0.6000000000000001, 'cardiovascular': 1.6, 'obesity': 0.6000000000000001, 'renal_chronic': 1.0, 'tobacco': 1.4000000000000001, 'contact_other_covid': 1.8, 'covid_res': 1.0, 'icu': 0.2}
    # weights = {'sex': 0.4, 'patient_type': 0.4, 'entry_date': 1.8, 'date_symptoms': 1.0, 'intubed': 1.0, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 2.0, 'diabetes': 1.0, 'copd': 0.8, 'asthma': 1.2000000000000002, 'inmsupr': 1.0, 'hypertension': 1.0, 'other_disease': 1.0, 'cardiovascular': 1.0, 'obesity': 1.0, 'renal_chronic': 1.0, 'tobacco': 1.0, 'contact_other_covid': 1.6, 'covid_res': 1.0, 'icu': 1.0}
    
    # 15 in 0.2 full set
    # weights = {'asthma': 1.8, 'icu': 0.6000000000000001, 'inmsupr': 0.4, 'age': 1.2000000000000002, 'contact_other_covid': 3.0, 'intubed': 1.0, 'pneumonia': 1.0, 'copd': 0.8, 'tobacco': 1.0, 'entry_date': 1.4000000000000001, 'date_symptoms': 1.8, 'hypertension': 1.0, 'other_disease': 0.6000000000000001, 'cardiovascular': 1.0, 'pregnancy': 1.2000000000000002, 'sex': 1.0, 'diabetes': 0.8, 'obesity': 1.2000000000000002, 'renal_chronic': 1.0, 'patient_type': 1.0, 'covid_res': 2.6}
    
    # 20 in 0.1 full set
    # weights = {'asthma': 1.8, 'icu': 0.5, 'inmsupr': 0.7000000000000001, 'age': 1.0, 'contact_other_covid': 1.9000000000000001, 'intubed': 1.0, 'pneumonia': 1.0, 'copd': 0.4, 'tobacco': 1.0, 'entry_date': 2.0, 'date_symptoms': 1.7000000000000002, 'hypertension': 1.0, 'other_disease': 2.0, 'cardiovascular': 0.1, 'pregnancy': 0.9, 'sex': 1.6, 'diabetes': 1.2000000000000002, 'obesity': 1.2000000000000002, 'renal_chronic': 0.8, 'patient_type': 1.0, 'covid_res': 1.1}
    
    # weights = {'icu': 0.5, 'inmsupr': 0.7000000000000001, 
    # 'copd': 0.4,  
    # 'cardiovascular': 0.1, 'pregnancy': 0.9, , 'renal_chronic': 0.8, 'patient_type': 1.0, }

    
    # weights = {}
    # weights = {'sex': 0.9, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.5, 'copd': 0.30000000000000004, 'asthma': 2.0, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.8, 'obesity': 1.0, 'renal_chronic': 1.0, 'tobacco': 0.7000000000000001, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}
    # weights = {'sex': 0.9, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.5, 'copd': 0.30000000000000004, 'asthma': 2.0, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.8, 'obesity': 1.0, 'renal_chronic': 1.0, 'tobacco': 0.7000000000000001, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}
    weights = {'sex': 0.7000000000000001, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.30000000000000004, 'copd': 0.30000000000000004, 'asthma': 2.2, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.4, 'obesity': 1.2000000000000002, 'renal_chronic': 1.0, 'tobacco': 0.8, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}


    # weights = {'other_disease': 0.9, 'entry_date': 1.8, 'contact_other_covid': 2.0, 'asthma': 0.9, 'date_symptoms': 1.5, 'sex': 1.8, 'diabetes': 0.8, 'obesity': 0.1, 'covid_res': 2.0, 'age': 1.2000000000000002, 'intubed': 0.6000000000000001, 'pneumonia': 1.0, 'tobacco': 0.30000000000000004, 'hypertension': 0.9, 'patient_type': 1.0, 'pregnancy': 0.5, 'renal_chronic': 0.9, 'inmsupr': 1.0, 'icu': 1.0, 'copd': 1.0, 'cardiovascular': 0.8}
    # weights = {'asthma': 1.8, 'icu': 0.5, 'inmsupr': 0.7000000000000001, 'age': 1.0, 'contact_other_covid': 2.7, 'intubed': 1.0, 'pneumonia': 1.0, 'copd': 0.30000000000000004, 'tobacco': 2.5, 'entry_date': 1.0, 'date_symptoms': 2.4000000000000004, 'hypertension': 0.5, 'other_disease': 1.5, 'cardiovascular': 0.30000000000000004, 'pregnancy': 1.0, 'sex': 1.1, 'diabetes': 1.0, 'obesity': 1.0, 'renal_chronic': 0.6000000000000001, 'patient_type': 1.0, 'covid_res': 2.0}
    # weights = {'age': 1.5, 'inmsupr': 0.8, 'renal_chronic': 1.7000000000000002, 'obesity': 1.0, 'diabetes': 1.0, 'hypertension': 1.0, 'asthma': 1.0, 'icu': 0.8, 'pneumonia': 1.0, 'cardiovascular': 1.2000000000000002, 'copd': 0.2, 'pregnancy': 1.1, 'sex': 0.8, 'intubed': 1.0, 'other_disease': 1.1, 'tobacco': 0.8, 'date_symptoms': 2.1, 'entry_date': 1.0, 'covid_res': 2.2, 'contact_other_covid': 2.9000000000000004, 'patient_type': 1.0}


    nbc = NBC(train, valid, weights)

    # start = time.time()

    # validClass = df['date_died'].tolist()
    # validClass = test_df['date_died'].tolist()
    
    # original ordering, bash 4, 30, 88.62, 88.49
    # bash 7, 50
    # bash 8, with weights
    # cols = ['sex', 'patient_type', 'entry_date', 'date_symptoms', 'intubed', 'pneumonia', \
    #             'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
    #             'hypertension', 'other_disease', 'cardiovascular', \
    #             'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
    #             'covid_res', 'icu']
    # weights = {'sex': 0.9, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.5, 'copd': 0.30000000000000004, 'asthma': 2.0, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.8, 'obesity': 1.0, 'renal_chronic': 1.0, 'tobacco': 0.7000000000000001, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}
    # weights = {'sex': 0.7000000000000001, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.30000000000000004, 'copd': 0.30000000000000004, 'asthma': 2.2, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.4, 'obesity': 1.2000000000000002, 'renal_chronic': 1.0, 'tobacco': 0.8, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}


    # ordered by max weight, 20, bash 3, 88.62
    # cols = ['other_disease', 'entry_date', 'contact_other_covid', 'asthma', 'date_symptoms', 'sex', 'diabetes', 'obesity', 'covid_res', \
    #         'age', 'intubed', 'pneumonia', 'tobacco', 'hypertension', 'patient_type', \
    #         'pregnancy', 'renal_chronic', 'inmsupr', 'icu', 'copd', 'cardiovascular']
    # weights = {'other_disease': 0.9, 'entry_date': 1.8, 'contact_other_covid': 2.0, 'asthma': 0.9, 'date_symptoms': 1.5, 'sex': 1.8, 'diabetes': 0.8, 'obesity': 0.1, 'covid_res': 2.0, 'age': 1.2000000000000002, 'intubed': 0.6000000000000001, 'pneumonia': 1.0, 'tobacco': 0.30000000000000004, 'hypertension': 0.9, 'patient_type': 1.0, 'pregnancy': 0.5, 'renal_chronic': 0.9, 'inmsupr': 1.0, 'icu': 1.0, 'copd': 1.0, 'cardiovascular': 0.8}

    # intuitive order, 30, bash 2, 88.58
    # cols = ['asthma', 'icu', 'inmsupr', 'age', 'contact_other_covid', 'intubed', \
    #         'pneumonia', 'copd', 'tobacco', 'entry_date', 'date_symptoms', 'hypertension', 'other_disease', \
    #         'cardiovascular', 'pregnancy', 'sex', 'diabetes', 'obesity', 'renal_chronic', 'patient_type', 'covid_res']
    #
    # weights = {'asthma': 1.8, 'icu': 0.5, 'inmsupr': 0.7000000000000001, 'age': 1.0, 'contact_other_covid': 2.7, 'intubed': 1.0, 'pneumonia': 1.0, 'copd': 0.30000000000000004, 'tobacco': 2.5, 'entry_date': 1.0, 'date_symptoms': 2.4000000000000004, 'hypertension': 0.5, 'other_disease': 1.5, 'cardiovascular': 0.30000000000000004, 'pregnancy': 1.0, 'sex': 1.1, 'diabetes': 1.0, 'obesity': 1.0, 'renal_chronic': 0.6000000000000001, 'patient_type': 1.0, 'covid_res': 2.0}

    # intuitive order 2, 30, bash 1, 88.63
    # cols = ['age', 'inmsupr', 'renal_chronic', 'obesity', 'diabetes', 'hypertension', \
    #         'asthma', 'icu', 'pneumonia', 'cardiovascular', 'copd', \
    #         'pregnancy', 'sex', 'intubed', \
    #         'other_disease', 'tobacco', 'date_symptoms', 'entry_date', \
    #         'covid_res', 'contact_other_covid', 'patient_type']
    #
    # weights = {'age': 1.5, 'inmsupr': 0.8, 'renal_chronic': 1.7000000000000002, 'obesity': 1.0, 'diabetes': 1.0, 'hypertension': 1.0, 'asthma': 1.0, 'icu': 0.8, 'pneumonia': 1.0, 'cardiovascular': 1.2000000000000002, 'copd': 0.2, 'pregnancy': 1.1, 'sex': 0.8, 'intubed': 1.0, 'other_disease': 1.1, 'tobacco': 0.8, 'date_symptoms': 2.1, 'entry_date': 1.0, 'covid_res': 2.2, 'contact_other_covid': 2.9000000000000004, 'patient_type': 1.0}

    '''
    for col in cols:
        bestW = 1
        mostAcc = 0
    
        for w in (np.arange(30)+1) * 0.1:
            nbc.weights[col] = w

            resultClass = []

            for i in range(len(df)):
                df_cols = df.columns.tolist()
                df_vals = df.iloc[i].tolist()
                pt = list(zip(df_cols, df_vals))

                resultClass.append(nbc.classifier(pt))
                      

            correct = np.sum(np.equal(resultClass, validClass))
            total = len(validClass)
            acc = correct/total

            if acc > mostAcc:
                mostAcc = acc
                bestW = w
        
        nbc.weights[col] = bestW

        print("col:", col, ", acc:", mostAcc, ", elapsed time =", time.time()-start)


    print("weights:", weights)
    
    '''

    
  
    resultClass = []

    for i in range(len(df)):
        cols = df.columns.tolist()
        vals = df.iloc[i].tolist()
        pt = list(zip(cols, vals))

        resultClass.append(nbc.classifier(pt))

    validClass = df['date_died'].tolist()

    correct = np.sum(np.equal(resultClass, validClass))
    total = len(validClass)

    print(*resultClass, sep = "\n")
    # print("overall accuracy:", correct/total)
    
    

    
  


def preprocess(filename):

    df = pd.read_csv(filename)

    df['entry_date'] = [dt.strptime(date, '%d-%m-%Y').toordinal() if '-' in date else dt.strptime(date, '%d/%m/%Y').toordinal() for date in df['entry_date'].tolist()]
    df['date_symptoms'] = [dt.strptime(date, '%d-%m-%Y').toordinal() if '-' in date else dt.strptime(date, '%d/%m/%Y').toordinal() for date in df['date_symptoms'].tolist()]

    df.loc[df['date_died'] != '9999-99-99', 'date_died'] = 1
    df.loc[df['date_died'] == '9999-99-99', 'date_died'] = 0

    for col in ['sex', 'patient_type', 'intubed', 'pneumonia', \
                'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
                'hypertension', 'other_disease', 'cardiovascular', \
                'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
                'covid_res', 'icu']:

        df.loc[np.logical_and(df[col] != 1, df[col] != 2), col] = 0

    return df
        

class NBC:

    def __init__(self, train, valid, weights):
        self.train = train
        self.valid = valid
        self.Dict = {}
        self.Dict['survived'] = {}
        self.Dict['died'] = {}
        self.CSVtoDict()
        self.weights = weights
        


    def CSVtoDict(self):

        df = preprocess(self.train)

        for col in df:
            if col not in ['entry_date', 'date_symptoms', 'age']:
                for val in df[col].unique():
                    self.Dict['died'][(col, val)] = sum(np.logical_and(df[col] == val, df['date_died'] == 1))
                    self.Dict['survived'][(col, val)] = sum(np.logical_and(df[col] == val, df['date_died'] == 0))

        self.Dict['survived']['age'] = norm(np.mean(df.loc[df['date_died'] == 0]['age'].tolist()), np.std(df.loc[df['date_died'] == 0]['age'].tolist()))
       
        a, loc, scale = skewnorm.fit(df.loc[df['date_died'] == 0]['entry_date'].tolist())
        self.Dict['survived']['entry_date'] = skewnorm(a, loc, scale)

        a, loc, scale = skewnorm.fit(df.loc[df['date_died'] == 0]['date_symptoms'].tolist())
        self.Dict['survived']['date_symptoms'] = skewnorm(a, loc, scale)

        self.Dict['died']['age'] = norm(np.mean(df.loc[df['date_died'] == 1]['age'].tolist()), np.std(df.loc[df['date_died'] == 1]['age'].tolist()))

        a, loc, scale = skewnorm.fit(df.loc[df['date_died'] == 1]['entry_date'].tolist())
        self.Dict['died']['entry_date'] = skewnorm(a, loc, scale)

        a, loc, scale = skewnorm.fit(df.loc[df['date_died'] == 1]['date_symptoms'].tolist())
        self.Dict['died']['date_symptoms'] = skewnorm(a, loc, scale)
   


    # class probability
    # returns probability of label
    def classP(self, classVal):
        totalSurvived = self.Dict['survived']['date_died', 0]
        totalDied = self.Dict['died']['date_died', 1]
        totalPatients = totalSurvived + totalDied

        pSurvived = totalSurvived / totalPatients
        pDied = totalDied / totalPatients

        if(classVal == 'survived'):
            return pSurvived
        else:
            return pDied


    # class count
    # returns count of patients of given class label
    def classCount(self, classVal):
        totalSurvived = self.Dict['survived']['date_died', 0]
        totalDied = self.Dict['died']['date_died', 1]
    
        if(classVal == 'survived'):
            return totalSurvived
        else:
            return totalDied


    # conditional probability for a feature given a class label
    def condP(self, feature, featureVal, classVal):
        # x is feature and its value
        # y is class
        if feature in ['age', 'entry_date', 'date_symptoms']:
            p = self.Dict[classVal][feature].pdf(featureVal)
        
        else:
            if feature in ['sex', 'patient_type', 'intubed', 'pneumonia', \
                    'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
                    'hypertension', 'other_disease', 'cardiovascular', \
                    'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
                    'covid_res', 'icu']:
                if featureVal == 0:
                    return 1

            y = self.classCount(classVal)
            x = self.Dict[classVal][feature, featureVal]
            p = x/y

        if feature in self.weights:
            p = p**self.weights[feature]

        return p


    # naive bayes classifier
    # classifies given x, an array/list of (feature, feature value)
    def classifier(self, x):
        pSurvivedGivenX = self.classP('survived')
        for feature, featureVal in x:
            if feature not in ['date_died']:
                pSurvivedGivenX *= self.condP(feature, featureVal, 'survived')


        pDiedGivenX = self.classP('died')
        for feature, featureVal in x:
            if feature not in ['date_died']:
                pDiedGivenX *= self.condP(feature, featureVal, 'died')

        pSurvivedGivenXNormalized = pSurvivedGivenX/(pSurvivedGivenX + pDiedGivenX)
        pDiedGivenXNormalized = pDiedGivenX/(pSurvivedGivenX + pDiedGivenX)


        if(pSurvivedGivenXNormalized > pDiedGivenXNormalized):
            # print("Patient will survive with a ", pSurvivedGivenXNormalized * 100, "% percent chance.")
            classification = 0

        elif(pSurvivedGivenXNormalized < pDiedGivenXNormalized):
            # print("Patient will die with a ", pDiedGivenXNormalized * 100, "% percent chance.")
            classification = 1

        else:
            # print("Patient has an equal chance of living and dying.")
            classification = random.randint(0, 1)

        
        return classification




if __name__ == '__main__':
    main()