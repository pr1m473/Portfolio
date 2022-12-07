import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from seaborn import heatmap

class Adaboost:
    def __init__(self,df,k,test_ratio=0.2,shuffle=False,weak_model='tree'):
        self.x_train, self.x_test , self.y_train, self.y_test =\
            train_test_split(np.array(df.iloc[:,:-1]),np.array(df.iloc[:,-1]),test_size=test_ratio,shuffle=shuffle)
        self.weights = np.ones((len(self.x_train)))*(1/len(self.x_train))
        self.weak_model = {'tree':self._tree_weak_learner(),
                            'naive':self._naivebayes_weak_learner(),
                            'SVM':self._svm_weak_learner()}
        self.weak_model = self.weak_model[weak_model]
        self.model_list = []
        self.k = k
        self.stage_values = []

    def _tree_weak_learner(self):
        weak_model = DecisionTreeClassifier(criterion='gini',max_depth=1)
        weak_model.fit(self.x_train,self.y_train,sample_weight=self.weights)
        return weak_model, weak_model.predict(self.x_train)

    def _naivebayes_weak_learner(self):
        weak_model = GaussianNB()
        weak_model.fit(self.x_train,self.y_train,sample_weight=self.weights)
        return weak_model, weak_model.predict(self.x_train)

    def _svm_weak_learner(self):
        weak_model = SVC()
        weak_model.fit(self.x_train,self.y_train,sample_weight=self.weights)
        return weak_model, weak_model.predict(self.x_train)


    def fit(self):
        for i in range(self.k):
            prediction = None
            weak_model = None
            weak_model, prediction = self.weak_model
            self.model_list.append(weak_model)
            t_error = (prediction != self.y_train)*1
            error = sum(self.weights * t_error)/sum(self.weights)
            stage = np.log((1-error)/error)
            self.stage_values.append(stage)
            self.weights *= np.exp(stage*t_error)

    def predict(self,pred_data):
        weighted_pred = np.zeros((len(pred_data)))
        for i in range(self.k):
            weighted_pred += self.model_list[i].predict(pred_data) * self.stage_values[i]
        return (weighted_pred > 0)*2 - 1

    def evaluate(self):
        predict = self.predict(self.x_test)
        accuracy = np.sum(predict == self.y_test)/len(predict) * 100
        sklearn_adaboost = AdaBoostClassifier(n_estimators=self.k)
        sklearn_adaboost.fit(self.x_train,self.y_train)
        sk_score = sklearn_adaboost.score(self.x_test,self.y_test)*100

        print(f"The accuracy of your Adaboost model is: {np.round(accuracy,2)}%")
        print(f"The accuracy of sklearn Adaboost model is: {np.round(sk_score, 2)}%")



if __name__=='__main__':
    hd_dataset =  pd.read_csv(r'G:\My Drive\Primerose 18\Adaboost\heart_disease_uci.csv')
    hd_dataset.isna().sum()
    hd_dataset.drop(['ca','thal','slope'],axis=1,inplace=True) # Check later if slope should be inserted
    (hd_dataset.isna().sum(axis=1)>3).sum()
    hd_dataset.dropna(thresh=11, axis=0, inplace=True)
    hd_dataset.isna().sum()
    hd_dataset['fbs'].fillna(hd_dataset['fbs'].mode()[0],inplace=True)
    hd_dataset['restecg'].fillna(hd_dataset['restecg'].mode()[0], inplace=True)
    # hd_dataset.drop(['sex', 'dataset','cp','restecg'], axis=1, inplace=True)
    hd_dataset[['trestbps', 'chol','oldpeak']] = hd_dataset[['trestbps', 'chol','oldpeak']].fillna(hd_dataset[['trestbps', 'chol','oldpeak']].median())

    hd_dataset['sex'] = pd.get_dummies(hd_dataset['sex'],drop_first=True)
    hd_dataset.rename({'sex':'male'},inplace=True,axis=1)
    dataset_onehot = pd.get_dummies(hd_dataset['dataset'])
    cp_onehot = pd.get_dummies(hd_dataset['cp'])
    hd_dataset['fbs'] = pd.get_dummies(hd_dataset['fbs'], drop_first=True)
    restecg_onehot = pd.get_dummies(hd_dataset['restecg'])
    # slope_onehot = pd.get_dummies(hd_dataset['slope'])
    # thal_onehot = pd.get_dummies(hd_dataset['thal'])
    hd_dataset['exang'] = pd.get_dummies(hd_dataset['exang'], drop_first=True)
    df_onehot =  pd.concat((hd_dataset.iloc[:,[1,2,5,6,7,9,10,11]],dataset_onehot,
               cp_onehot,restecg_onehot),axis=1)
    correlation = df_onehot.corr()
    df_onehot.isna().sum()

    df_onehot['label'] = hd_dataset['num'].apply(lambda x : 1 if x>0 else -1)

    adaboost_model1 = Adaboost(df_onehot,4,shuffle=True)
    adaboost_model1.fit()
    adaboost_model1.evaluate()
















