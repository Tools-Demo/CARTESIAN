import pandas as pd
import numpy as np
from sklearn import preprocessing
from datetime import datetime
import pickle
import math


df_r = pd.read_csv("Dataset/response.csv", sep=",", encoding="ISO-8859-1")

project_list = ['react', 'django', 'nixpkgs', 'scikit-learn', 'yii2', 'cdnjs', 'terraform', 'cmssw', 'salt', 'tensorflow', 'pandas',
                'symfony', 'moby', 'rails', 'rust', 'kubernetes'
                 ]

cartersian_result_folder_path = 'CARTESIAN_Results/'



# One hot encoding
def encode_labels(df1, column_name):
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return  one_hot_vector


df_r['Language'] = encode_labels(df_r, 'Language')
df_r['Project_Domain'] = encode_labels(df_r, 'Project_Domain')



predictors_a = ['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
                   'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency', 'Comments_Per_Closed_PR',
                   'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Project_Domain', 'Additions', 'Deletions',
                   'Wait_Time', 'PR_Latency', 'Files_Changed', 'Label_Count', 'Workload', 'Commits_Average', 'Contributor',
                   'Followers', 'Closed_Num', 'Public_Repos', 'Accept_Num', 'User_Accept_Rate', 'Contributions', 'Closed_Num_Rate',
                   'Prev_PRs', 'Open_Issues', 'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2',
                   'X1_3', 'X1_4', 'X1_5','X1_6', 'X1_7', 'X1_8', 'X1_9']
predictors_r = ['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size', 'Additions_Per_Week',
                     'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday', 'Monday',
                     'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR',
                     'Forks_Count','File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch', 'Project_Domain',
                     'Additions', 'Deletions', 'Day', 'Commits_PR', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency', 'Files_Changed',
                     'Label_Count', 'Assignees_Count', 'Workload', 'PR_age', 'Commits_Average', 'Contributor', 'Followers',
                     'Closed_Num', 'Public_Repos', 'Organization_Core_Member', 'Accept_Num', 'User_Accept_Rate', 'Contributions',
                     'Closed_Num_Rate', 'Following', 'Prev_PRs', 'Review_Comments_Count', 'Participants_Count', 'Comments_Count',
                     'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues', 'first_response', 'latency_after_first_response',
                     'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9']


start_date = '2017-09-01'
end_date = '2018-01-31'

target_a = 'PR_accept'
target_r = 'PR_response'


X_test = df_r.loc[(df_r['PR_Date_Created_At'] >= start_date) & (df_r['PR_Date_Created_At'] <= end_date)]




def CARTESIAN_Model(df_test_PR, folder_path):
    pd.options.mode.chained_assignment = None
    with open('Trained_models/accept_XGB.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        # y_pred_accept = accept_model.predict(df_test_PR[predictors_a])
        y_pred_accept = accept_model.predict_proba(df_test_PR[predictors_a])[:,1]
        df_test_PR['Result_Accept'] = y_pred_accept

    with open('Trained_models/response_XGB.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        # y_pred_response = response_model.predict(df_test_PR[predictors_r])
        y_pred_response = response_model.predict_proba(df_test_PR[predictors_r])[:,1]
        df_test_PR['Result_Response'] = y_pred_response

    df_test_PR['Score'] = df_test_PR['Result_Accept'].apply(np.exp) + df_test_PR['Result_Response'].apply(np.exp)
    print(df_test_PR[['Pull_Request_ID', 'Result_Accept', 'Result_Response', 'Score']].head(10))
    df_test_PR.to_csv(cartersian_result_folder_path+'cartersian_results.csv', sep=',', encoding='utf-8', index=False)
    return df_test_PR



def get_top_n_MAP(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30', '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31'
                     }
    top_k_list = [5, 10, 20]
    df = df.set_index('Project_Name')
    df_algo_results = pd.DataFrame(columns=['Project', 'MAP_5'])
    for top_k in top_k_list:
        print("Now calculating MAP for {}".format(top_k))
        df_MAP = pd.DataFrame(columns=['MAP_'+str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                for i in range(1, last_date+1):
                    MAP_accept_response = 0
                    positive_count_accept_response = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if counter >top_k: break
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                                positive_count_accept_response += 1
                                MAP_accept_response += positive_count_accept_response/(counter)

                    total_result.append(MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0)
                    MAP_day = MAP_accept_response/positive_count_accept_response if positive_count_accept_response !=0 else 0
            if top_k > 5:
                df_MAP = df_MAP.append({'MAP_'+str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_MAP.shape)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'MAP_' + str(top_k): np.mean(total_result)}, ignore_index=True)
                print(df_algo_results.shape)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_MAP['MAP_'+str(top_k)]], axis=1)
    print(df_algo_results.head())
    df_algo_results.sort_values(by=['MAP_5', 'MAP_10', 'MAP_20'], ascending=False).to_csv(
        folder_path+'MAP_all_results.csv', sep=',', encoding='utf-8', index=False)


def get_top_n_recall(df, folder_path):
    months_tested = {'2017-09-01': '2017-09-30', '2017-10-01': '2017-10-31',
                     '2017-11-01': '2017-11-30', '2017-12-01': '2017-12-31',
                     '2018-01-01': '2018-01-31',
                     }
    df = df.set_index('Project_Name')
    top_k_list = [5, 10, 20]
    df_algo_results = pd.DataFrame(columns=['Project', 'AR_5'])
    for top_k in top_k_list:
        print("Now calculating AR for {}".format(top_k))
        df_AR = pd.DataFrame(columns=['AR_' + str(top_k)])
        for project in project_list:
            total_result = []
            df_project = df.loc[[project]]
            for date_start, date_end in months_tested.items():
                df_month = df_project.loc[
                    (df_project['PR_Date_Created_At'] >= date_start) & (df_project['PR_Date_Created_At'] <= date_end)]
                print(project)
                last_date = datetime.strptime(date_end, "%Y-%m-%d").day
                for i in range(1, last_date+1):
                    top_recall_num = 0
                    total_recall_num = 0
                    counter = 0
                    for index, row in df_month.iterrows():
                        if datetime.strptime(row['PR_Date_Created_At'], "%Y-%m-%d").day == i:
                            counter += 1
                            if row['PR_accept'] == 1 and row['PR_response'] == 1:
                                if counter <= top_k:
                                    top_recall_num += 1
                                total_recall_num += 1
                    total_result.append(top_recall_num/total_recall_num if total_recall_num !=0 else 0)
                    AR_day = top_recall_num/total_recall_num if total_recall_num !=0 else 0
                print('Project {} in month {} have average recall {}'.format(
                    project, date_start.split('-')[0] + '-' + date_start.split('-')[1], AR_day))

            if top_k > 5:
                df_AR = df_AR.append({'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
            else:
                df_algo_results = df_algo_results.append(
                    {'Project': project, 'AR_' + str(top_k): np.mean(total_result)}, ignore_index=True)
        if top_k > 5:
            df_algo_results = pd.concat([df_algo_results, df_AR['AR_' + str(top_k)]], axis=1)
    df_algo_results.sort_values(by=['AR_5', 'AR_10', 'AR_20'], ascending=False).to_csv(
        folder_path+'AR_all_results.csv', sep=',', encoding='utf-8', index=False)





if __name__ == '__main__':

    df = CARTESIAN_Model(X_test, cartersian_result_folder_path)
    get_top_n_MAP(df.sort_values(by=['Score'], ascending=False), cartersian_result_folder_path)
    get_top_n_recall(df.sort_values(by=['Score'], ascending=False), cartersian_result_folder_path)




