import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from statistics import mean
import xgboost as xgb
import pickle


df = pd.read_csv("/home/ppf/Ilyas_dataset/accept.csv", sep=",", encoding="ISO-8859-1")


project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes'
                 ]


# Remove some of the PRs with negative latency
ID = df.Pull_Request_ID[df.latency_after_first_response < 0]
df = df.loc[~df.Pull_Request_ID.isin(ID)]


scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']
results = pd.DataFrame(columns=['Model', 'Project', 'AUC', 'Precision', 'Recall', 'F-measure', 'Accuracy'])


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



def modelfit(clf, xtrain, ytrain, xtest, ytest, results, project, early_stopping_rounds=50):
    clf = clf.fit(xtrain, ytrain, verbose=11)
    y_pred = clf.predict(xtest)
    y_pred_proba = clf.predict_proba(xtest)[:,1]


    print('Test error: {:.3f}'.format(1- metrics.accuracy_score(ytest, y_pred)))
    # # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(ytest, y_pred))
    print("AUC Score : %f" % metrics.roc_auc_score(ytest, y_pred_proba))
    print("Recall : %f" % metrics.recall_score(ytest, y_pred))
    print("Precision : %f" % metrics.precision_score(ytest, y_pred))
    print("F-measure : %f" % metrics.f1_score(ytest, y_pred))
    return results.append(
        {'Model': 'XGBoost', 'Project': project,  'AUC': metrics.roc_auc_score(ytest, y_pred_proba),
         'Precision': metrics.precision_score(ytest, y_pred),
         'Recall': metrics.recall_score(ytest, y_pred),
         'F-measure': metrics.f1_score(ytest, y_pred),
         'Accuracy': metrics.accuracy_score(ytest, y_pred)},
        ignore_index=True)


df = df.set_index('Project_Name')
classifiers = get_classifiers()

for name, value in classifiers.items():
    for project in project_list:
        df_proj = df.loc[[project]]
        print(project+' has a size of {}'.format(df_proj.shape))
        def encode_labels(df1, column_name):
            encoder = preprocessing.LabelEncoder()
            df1[column_name] = [str(label) for label in df1[column_name]]
            encoder.fit(df1[column_name])
            one_hot_vector = encoder.transform(df1[column_name])
            return one_hot_vector


        df_proj['Language'] = encode_labels(df_proj, 'Language')
        df_proj['Project_Domain'] = encode_labels(df_proj, 'Project_Domain')

        df_proj = df_proj[['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
         'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Comments_Per_Closed_PR',
         'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Project_Domain', 'Additions', 'Deletions',
         'Wait_Time', 'PR_Latency', 'Files_Changed', 'Label_Count', 'Workload', 'Commits_Average', 'Contributor',
         'Followers', 'Closed_Num', 'Public_Repos', 'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate',
         'Prev_PRs', 'Open_Issues', 'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5',
          'X1_6', 'X1_7', 'X1_8', 'X1_9',
          'PR_accept', 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At', 'PR_Time_Closed_At']]

        target = 'PR_accept'
        start_date = '2017-09-01'
        end_date = '2018-02-28'

        X_test = df_proj.loc[(df_proj['PR_Date_Created_At'] >= start_date) & (df_proj['PR_Date_Created_At'] <= end_date)]
        y_test = X_test[target]
        X_train = df_proj.loc[(df_proj['PR_Date_Created_At'] < start_date)]
        y_train = X_train[target]



        ## Select the predictors
        predictors = [x for x in df_proj.columns if
                      x not in [target, 'PR_Date_Created_At', 'PR_Time_Create_At', 'PR_Date_Closed_At',
                                'PR_Time_Closed_At']]

        X_train = X_train[predictors]
        X_test = X_test[predictors]

        # Scale the training dataset
        scaler_train = preprocessing.StandardScaler()
        X_train_scaled = scaler_train.fit_transform(np.array(X_train[predictors]).astype('float64'))
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=predictors)

        scaler_test = preprocessing.StandardScaler()
        X_test_scaled = scaler_test.fit_transform(np.array(X_test[predictors]).astype('float64'))
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=predictors)

        clf = value
        print('Classifer: ', name)
        if name == 'XGBoost':
            clf = xgb.XGBClassifier(**params)
            results = modelfit(clf, X_train, y_train, X_test, y_test, results, project)
        else:
            if name == 'LinearSVC':
                clf.fit(X_train, y_train)
                clf = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
                y_predprob = clf.predict_proba(X_test_scaled)[:,1]
            else:
                if name == 'LogisticRegression':
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    y_predprob = clf.predict_proba(X_test_scaled)[:,1]
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    y_predprob = clf.predict_proba(X_test)[:, 1]

            ## Print model report:
            print("\nModel Report")
            print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_predprob))
            print("Recall : %f" % metrics.recall_score(y_test, y_pred))
            print("Precision : %f" % metrics.precision_score(y_test, y_pred))
            print("F-measure : %f" % metrics.f1_score(y_test, y_pred))
            results= results.append(
                {'Model': name, 'Project': project, 'AUC': metrics.roc_auc_score(y_test, y_predprob),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F-measure': metrics.f1_score(y_test, y_pred),
                 'Accuracy': metrics.accuracy_score(y_test, y_pred)},
                ignore_index=True)



results.to_csv('Results/accept_projects_results.csv', sep=',', encoding='utf-8', index=False)

