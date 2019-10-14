import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from statistics import mean
import xgboost as xgb
from xgboost import plot_importance
import pickle
import seaborn as sns
sns.set(style="white")
sns.set(style='whitegrid', color_codes=True)

df = pd.read_csv("/home/ppf/Ilyas_dataset/response.csv", sep=",", encoding="ISO-8859-1")

# 'cmssw'
project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes'
                 ]


# Remove some of the PRs with negative latency
ID = df.Pull_Request_ID[df.latency_after_first_response < 0]
df = df.loc[~df.Pull_Request_ID.isin(ID)]


scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
results = pd.DataFrame(columns=['Model', 'AUC', 'Precision', 'Recall', 'F-measure', 'Accuracy'])


def get_classifiers():
    return {
        'RandomForest': RandomForestClassifier(n_jobs=4, n_estimators=200, bootstrap=True, class_weight='balanced',
                                                random_state=42, oob_score=True), 
        'LinearSVC': LinearSVC(max_iter=1500),
        'LogisticRegression': LogisticRegression(solver='lbfgs', n_jobs=4, multi_class='auto', max_iter=1200),
        'XGBoost':'XGBoost',
    }


params = {
			'objective': 'binary:logistic',
			'eta': 0.08,
			'colsample_bytree': 0.886,
			'min_child_weight': 1.1,
			'max_depth': 7,
			'subsample': 0.886,
			'gamma': 0.1,
			'lambda':10,
			'verbose_eval': True,
			'eval_metric': 'auc',
			'scale_pos_weight':6,
			'seed': 201703,
			'missing':-1
	   }


def encode_labels(df1, column_name):
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return one_hot_vector


df['Language'] = encode_labels(df, 'Language')
df['Project_Domain'] = encode_labels(df, 'Project_Domain')

df = df[['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
     'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday', 'Monday',
     'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR', 'Forks_Count',
     'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch', #'Project_Domain',#'Mergeable',
     'Additions', 'Deletions', 'Day', 'Commits_PR', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency', 'Files_Changed',
     'Label_Count', 'Assignees_Count', 'Workload', 'PR_age', #'PR_accept',
     'Commits_Average', 'Contributor', 'Followers', 'Closed_Num', 'Public_Repos', 'Organization_Core_Member',#'wait_time_up',
     'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate', 'Following', 'Prev_PRs', 'Review_Comments_Count',
     'Participants_Count', 'Comments_Count', 'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues',
     'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9',
      'PR_response', 'PR_Date_Created_At',
     'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At', 'Project_Name']]


"""
    Total Train dataset size: (596323, 75)
    Total Test dataset size: (81394, 75)
    Balance of the dataset
    Number of responded pull requests (164643, 75)
    Number of unresponded pull requests (431680, 75)
    Balanced Dataset size: (342996, 74)

"""


target = 'PR_response'
start_date = '2017-09-01'
end_date = '2018-02-28'


X_test = df.loc[(df['PR_Date_Created_At'] >= start_date) & (df['PR_Date_Created_At'] <= end_date)]
y_test = X_test[target]
X_train = df.loc[(df['PR_Date_Created_At'] < start_date)]
y_train = X_train[target]



## Select the predictors
predictors = [x for x in df.columns if x not in [target, 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
                        'PR_Time_Closed_At', 'Project_Name']]

X_train = X_train[predictors]
X_test = X_test[predictors]


# Scale the training dataset
scaler_train = preprocessing.StandardScaler()
X_train_scaled = scaler_train.fit_transform(np.array(X_train[predictors]).astype('float64'))
X_train_scaled = pd.DataFrame(X_train_scaled, columns=predictors)

scaler_test = preprocessing.StandardScaler()
X_test_scaled = scaler_test.fit_transform(np.array(X_test[predictors]).astype('float64'))
X_test_scaled = pd.DataFrame(X_test_scaled, columns=predictors)


def modelfit(clf, xtrain, ytrain, xtest, ytest, results, early_stopping_rounds=50):
    clf = clf.fit(xtrain, ytrain, verbose=11)
    y_pred = clf.predict(xtest)
    y_pred_proba = clf.predict_proba(xtest)[:,1]

    print('Test error: {:.3f}'.format(1- metrics.accuracy_score(ytest, y_pred)))

    # # Save the model
    with open('Models/response_XGB.pickle.dat', 'wb') as f:
         pickle.dump(clf, f)

    # # Load the model
    # # with open('response_xgb_16.pickle.dat', 'rb') as f:
    # #     load_xgb = pickle.load(f)
    #
    # # y_pred = load_xgb.predict(dtest[predictors])
    # # dtest_predproba = load_xgb.predict_proba(dtest[predictors])[:,1]
    #
    # y_pred = alg.predict(dtest[predictors])
    # dtest_predproba = alg.predict_proba(dtest[predictors])[:, 1]
    #
    # # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(ytest, y_pred))
    print("AUC Score : %f" % metrics.roc_auc_score(ytest, y_pred_proba))
    print("Recall : %f" % metrics.recall_score(ytest, y_pred))
    print("Precision : %f" % metrics.precision_score(ytest, y_pred))
    print("F-measure : %f" % metrics.f1_score(ytest, y_pred))
    return results.append(
        {'Model': 'XGBoost', 'AUC': metrics.roc_auc_score(ytest, y_pred_proba),
         'Precision': metrics.precision_score(ytest, y_pred),
         'Recall': metrics.recall_score(ytest, y_pred),
         'F-measure': metrics.f1_score(ytest, y_pred),
         'Accuracy': metrics.accuracy_score(ytest, y_pred)},
        ignore_index=True)



classifiers = get_classifiers()

for name, value in classifiers.items():
        clf = value
        print('Classifer: ', name)
        if name == 'XGBoost':
            clf = xgb.XGBClassifier(**params)
            results = modelfit(clf, X_train, y_train, X_test, y_test, results)
        else:
            if name == 'LinearSVC':
                clf.fit(X_train_scaled, y_train)
                clf = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                y_predprob = clf.predict_proba(X_test_scaled)[:,1]
                with open('Models/response_' + name + '.pickle.dat', 'wb') as f:
                    pickle.dump(clf, f)
            else:
                if name == 'LogisticRegression':
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    y_predprob = clf.predict_proba(X_test_scaled)[:,1]
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_predprob = clf.predict_proba(X_test)[:, 1]
            with open('Models/response_' + name + '.pickle.dat', 'wb') as f:
                pickle.dump(clf, f)
            ## Print model report:
            print("\nModel Report")
            print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            results= results.append(
                {'Model': name, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-measure': metrics.f1_score(y_test, y_pred),
                 'Accuracy': metrics.accuracy_score(y_test, y_pred)},
                ignore_index=True)



results.to_csv('Results/response_results.csv', sep=',', encoding='utf-8', index=False)


