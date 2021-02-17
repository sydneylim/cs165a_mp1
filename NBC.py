import numpy as np
import pandas as pd
import sys
from datetime import datetime as dt
from scipy.stats import norm, skewnorm
import random

def main():
    train = sys.argv[1]
    valid = sys.argv[2]

    df = preprocess(valid)

    nbc = NBC(train, valid)

    resultClass = []

    for i in range(len(df)):
        cols = df.columns.tolist()
        vals = df.iloc[i].tolist()
        pt = list(zip(cols, vals))

        # for col, val in pt:
        #     print(col, val)
        resultClass.append(nbc.classifier(pt))

    validClass = df['date_died'].tolist()

    correct = np.sum(np.equal(resultClass, validClass))
    total = len(validClass)

    print(*resultClass, sep = "\n")
    # print(correct/total)
    


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

    # df.loc[df['age'] <= 18, 'age'] = 1
    # df.loc[np.logical_and(df['age'] <= 30, df[col] > 18), 'age'] = 2
    # df.loc[np.logical_and(df['age'] <= 65, df[col] > 30), 'age'] = 3
    # df.loc[df['age'] > 65, 'age'] = 4

    return df
        

class NBC:

    def __init__(self, train, valid):
        self.train = train
        self.valid = valid
        self.Dict = {}
        self.Dict['survived'] = {}
        self.Dict['died'] = {}
        self.CSVtoDict()
        


    def CSVtoDict(self):

        df = preprocess(self.train)

        for col in df:
            if col not in ['entry_date', 'date_symptoms']:
                for val in df[col].unique():
                    self.Dict['died'][(col, val)] = sum(np.logical_and(df[col] == val, df['date_died'] == 1))
                    self.Dict['survived'][(col, val)] = sum(np.logical_and(df[col] == val, df['date_died'] == 0))

                    #Dict[(col, val)] = sum(df[col] == val)

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
        # for key in Dict:
        #     print(key,':', Dict['date_died'][key])


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
            return self.Dict[classVal][feature].pdf(featureVal)
        
        else:
            # if feature in ['sex', 'patient_type', 'intubed', 'pneumonia', \
            #         'pregnancy', 'diabetes', 'copd', 'asthma', 'inmsupr', \
            #         'hypertension', 'other_disease', 'cardiovascular', \
            #         'obesity', 'renal_chronic', 'tobacco', 'contact_other_covid', \
            #         'covid_res', 'icu']:
            #     if featureVal == 0:
            #         return 1

            y = self.classCount(classVal)
            x = self.Dict[classVal][feature, featureVal]
            return x/y


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