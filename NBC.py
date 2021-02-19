import numpy as np
import pandas as pd
import sys
from datetime import datetime as dt
from scipy.stats import norm, skewnorm
import random
import time

def main():

    # start = time.time()

    train = sys.argv[1]
    valid = sys.argv[2]


    df = (preprocess(valid))
    
    
    # weights = {}
    weights = {'sex': 0.9, 'patient_type': 0.30000000000000004, 'entry_date': 1.6, 'date_symptoms': 1.6, 'intubed': 0.9, 'pneumonia': 1.0, 'age': 1.0, 'pregnancy': 0.9, 'diabetes': 0.5, 'copd': 0.30000000000000004, 'asthma': 2.0, 'inmsupr': 2.6, 'hypertension': 1.0, 'other_disease': 1.4000000000000001, 'cardiovascular': 0.8, 'obesity': 1.0, 'renal_chronic': 1.0, 'tobacco': 0.7000000000000001, 'contact_other_covid': 1.5, 'covid_res': 1.0, 'icu': 1.0}

    nbc = NBC(train, valid, weights)

    # calculating the weights using the validation data set, offline
    '''
    validClass = df['date_died'].tolist()

    cols = ['sex', 'patient_type', 'entry_date', 'date_symptoms', 'intubed', 'pneumonia', \
                'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
                'hypertension', 'other_disease', 'cardiovascular', \
                'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
                'covid_res', 'icu']

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

    # # printing accuracies based on class
    # resultClass = np.array(resultClass)
    # validClass = np.array(validClass)

    # mask = validClass == 1
    # resultClassDied = resultClass[mask]
    # validClassDied = validClass[mask]

    # mask = validClass == 0
    # resultClassSurvived = resultClass[mask]
    # validClassSurvived = validClass[mask]
    
    # accDied = np.sum(resultClassDied == validClassDied)/ len(validClassDied)
    # accSurvived = np.sum(resultClassSurvived == validClassSurvived)/ len(validClassSurvived)
    # accTotal = np.sum(resultClass == validClass)/ len(validClass)

    # print("died accuracy:", accDied)
    # print("survived accuracy:", accSurvived)
    # print("total accuracy:", accTotal)

    print(*resultClass, sep = "\n")   

    # print(time.time() - start)


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
            classification = 0

        elif(pSurvivedGivenXNormalized < pDiedGivenXNormalized):
            classification = 1

        else:
            classification = random.randint(0, 1)

        
        return classification


if __name__ == '__main__':
    main()