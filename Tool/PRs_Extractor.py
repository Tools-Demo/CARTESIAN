#!/usr/bin/env python
# -*-coding:utf-8-*-

'''
    Author: pq
    Function: Features Extractor

'''
import logging
import json
import requests
import sys
import re
import gensim
import csv
import time
from datetime import datetime
import pandas as pd
import pickle
import multiprocessing
# Pyinstaller fix
multiprocessing.freeze_support()
from sklearn import preprocessing
import os
import xgboost
import numpy as np


model = gensim.models.Word2Vec.load('Models/word2vec_model')


log_file = open('Log_files/log.txt','w')
stdo=sys.stdout
ster=sys.stderr
sys.stderr=log_file
sys.stdout=log_file
logging.getLogger().setLevel(logging.INFO)
error_log = open('Log_files/error_log.txt','w')
# access_token= '74ae820b0e2d110af84821955088cb16f11c7222'


pr_extraction_completed = 0
feature_extraction_completed = 0

def flat(tree):
    res = []
    for i in tree:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def get_request_body(url, access_token, get_headers=None):
    body = []
    loop_url = url + '&access_token=' + access_token

    headers = {'Accept': 'application/vnd.github.v3.text+json'} if get_headers is None else get_headers
    error_message = None
    try:
        response = requests.get(loop_url, timeout=10, headers=headers)
        if response.status_code == requests.codes.ok:
            data = response.json()
            if isinstance(data, dict) and 'items' in data.keys():
                body.append(data['items'])
            elif isinstance(data, dict) and 'errors' in data.keys():
                error_message = 'Arguments Error:'+loop_url+'\n'
            else:
                body.append(data)
            if 'next' in response.links.keys():
                loop_url = response.links['next']['url']
            if int(response.headers['X-RateLimit-Remaining']) == 0:
                error_message = "Rate limit exceeded. Please try again later"

        elif str(response.status_code) == '404' or str(response.status_code) == '451':
            error_message = "Given repo not found. Please try with a valid repo name."
        elif str(response.status_code) == '403':
            error_message = "Maximum number of login attempts exceeded. Please try again later."
        else:
            error_message = 'Invalid access token. Please try with a valid access token.'
    except Exception as e:
        error_message = 'Unknown error occured, please try again later.'

    return flat(body), error_message


class Features:
    def __init__(self, token):
        self.token = token


    def check_links(self, project_fullname, token):
        self.token = token
        self.project, self.error_message = get_request_body(
            'https://api.github.com/repos/' + project_fullname + '?per_page=100&access_token=' + self.token, self.token)

        if self.error_message is not None:
            return self.error_message

        self.contributors, self.error_message = get_request_body(
            self.project[0]['contributors_url'] + '?anon=1&per_page=100&access_token=' + self.token, self.token)
        if self.error_message is not None:
            return self.error_message

        # if len(self.contributors) < 1000:
        #     return
        pulls, self.error_message = get_request_body('https://api.github.com/repos/' + self.project[0][
            'full_name'] + '/pulls?state=open&per_page=100&access_token=' + self.token, self.token)
        if self.error_message is not None:
            return self.error_message
        # self.pulls_details = []
        # for i in self.pulls:
        #    self.pulls_details.append(get_request_body(i['url']+'?access_token='+self.token)[0])
        self.issues = []
        issues_list, self.error_message = get_request_body('https://api.github.com/repos/{}/issues?per_page=100&access_token={}'.
                                                           format(self.project[0]['full_name'],self.token), self.token)
        if self.error_message is not None:
            return self.error_message
        for i in issues_list:
            tmp_dict = {}
            tmp_dict['created_at'] = i['created_at']
            tmp_dict['closed_at'] = i['closed_at']
            if 'pull_request' in i.keys():
                tmp_dict['pull_request'] = i['pull_request']
            self.issues.append(tmp_dict)
        self.members, self.error_message = get_request_body('https://api.github.com/orgs/{}/members?per_page=100&access_token={}'.format(
            self.project[0]['organization']['login'], self.token), self.token)
        if self.error_message is not None:
            return self.error_message

        Statistics1, self.error_message = get_request_body('https://api.github.com/repos/' + self.project[0][
            'full_name'] + '/stats/commit_activity?per_page=100&access_token=' + self.token, self.token)
        if self.error_message is not None:
            return self.error_message

        # Statistics
        Sunday = 0
        Monday = 0
        Tuesday = 0
        Wednesday = 0
        Thursday = 0
        Friday = 0
        Saturday = 0
        for i in Statistics1:
            Sunday += i['days'][0]
            Monday += i['days'][1]
            Tuesday += i['days'][2]
            Wednesday += i['days'][3]
            Thursday += i['days'][4]
            Friday += i['days'][5]
            Saturday += i['days'][6]
        Statistics2, self.error_message = get_request_body('https://api.github.com/repos/' + self.project[0][
            'full_name'] + '/stats/code_frequency?per_page=100&access_token=' + self.token, self.token)
        if self.error_message is not None:
            return self.error_message
        Additions_Per_Week = 0
        Deletions_Per_Week = 0
        for i in range(int(len(Statistics2) / 3)):
            Additions_Per_Week += Statistics2[i * 3 + 1]
            Deletions_Per_Week += Statistics2[i * 3 + 2]
        self.Statistics = {}
        self.Statistics['Sunday'] = Sunday / (len(Statistics1) + 1e-10)
        self.Statistics['Monday'] = Monday / (len(Statistics1) + 1e-10)
        self.Statistics['Tuesday'] = Tuesday / (len(Statistics1) + 1e-10)
        self.Statistics['Wednesday'] = Wednesday / (len(Statistics1) + 1e-10)
        self.Statistics['Thursday'] = Thursday / (len(Statistics1) + 1e-10)
        self.Statistics['Friday'] = Friday / (len(Statistics1) + 1e-10)
        self.Statistics['Saturday'] = Saturday / (len(Statistics1) + 1e-10)
        self.Statistics['Additions_Per_Week'] = Additions_Per_Week / (len(Statistics2) / 3 + 1e-10)
        self.Statistics['Deletions_Per_Week'] = Deletions_Per_Week / (len(Statistics2) / 3 + 1e-10)
        self.Statistics['Contributor_Num'] = len(self.contributors)
        self.Statistics['Language'] = self.project[0]['language']
        self.Statistics['Team_Size'] = len(self.members)
        self.Statistics['Watchers'] = self.project[0]['subscribers_count']
        self.Statistics['Stars'] = self.project[0]['watchers']
        self.pulls_details = []
        for i in pulls:
            tmp, self.error_message = get_request_body(i['url'] + '?access_token=' + self.token, self.token)
            if self.error_message is not None:
                return self.error_message
            tmp_dict = {}
            tmp_dict['user'] = {'type': tmp[0]['user']['type'], 'login': tmp[0]['user']['login'],
                                'repos_url': tmp[0]['user']['repos_url']}
            tmp_dict['closed_at'] = tmp[0]['closed_at']
            tmp_dict['author_association'] = tmp[0]['author_association']
            tmp_dict['created_at'] = tmp[0]['created_at']
            tmp_dict['merged_at'] = tmp[0]['merged_at']
            tmp_dict['deletions'] = tmp[0]['deletions']
            tmp_dict['comments'] = tmp[0]['comments']
            tmp_dict['changed_files'] = tmp[0]['changed_files']
            tmp_dict['commits'] = tmp[0]['commits']
            tmp_dict['title'] = tmp[0]['title'] if tmp[0]['title'] is not None else None
            tmp_dict['body'] = tmp[0]['body']
            tmp_dict['base'] = {'ref': tmp[0]['base']['ref']}
            tmp_dict['head'] = {'ref': tmp[0]['head']['ref']}
            tmp_dict['mergeable_state'] = tmp[0]['mergeable_state']
            tmp_dict['rebaseable'] = tmp[0]['rebaseable']
            tmp_dict['mergeable'] = tmp[0]['mergeable']
            tmp_dict['labels'] = tmp[0]['labels']
            tmp_dict['body_text'] = tmp[0]['body_text']
            tmp_dict['review_comments_url'] = tmp[0]['review_comments_url']
            tmp_dict['comments_url'] = tmp[0]['comments_url']
            tmp_dict['assignees'] = tmp[0]['assignees']
            tmp_dict['additions'] = tmp[0]['additions']
            tmp_dict['url'] = tmp[0]['url']
            tmp_dict['html_url'] = tmp[0]['html_url']
            tmp_dict['number'] = tmp[0]['number']
            # tmp_files = get_request_body(i['url'] + '/files' + '?access_token=' + self.token)[0]
            # tmp_files_list = []
            # for f in tmp_files:
            #     file_dict = {}
            #     file_dict['filename'] = f['filename']
            #     file_dict['raw_url'] = f['raw_url']
            #     file_dict['blob_url'] = f['blob_url']
            #     file_dict['patch'] = f['patch']
            #     tmp_files_list.append(file_dict)
            # tmp_dict['pr_files_details'] = tmp_files_list
            self.pulls_details.append(tmp_dict)
        self.user_issues_all = {}
        self.user_issues_merged = {}
        self.forks = []
        forks_list, self.error_message = get_request_body('https://api.github.com/repos/' + self.project[0][
            'full_name'] + '/forks?per_page=100&access_token=' + self.token, self.token)
        if self.error_message is not None:
            return self.error_message
        for i in forks_list:
            self.forks.append({'created_at': i['created_at']})

        self.user_project = {}

        return self.error_message


    def get_pull_request_features(self):
        if len(self.contributors) < 1:
            return
        for i in self.pulls_details:
            if i['user']['type'] == 'User':
                yield self.getFeatures(i)

    def getFeatures(self, pull_request):
        end_time = time.strptime(pull_request['created_at'], "%Y-%m-%dT%H:%M:%SZ")

        def user_features(pull_request):
            '''
                User Features:
                #Contribution Rate---the percentage of commits by the author currently in the project  (Search)
                Accept Rate---the percentage of the author's other PRs that have been merged(Search-->issue)
                Close_Rate---the percentage of the author's other PRs that have been closed
                Prev_PRs---Number of pull requests submitted by a specific developer,prior to the examined one
                Followers---followers to the developer at creation(没有考虑at creation)
                Following---the number of following(没有考虑at creation)
                public_repos --- the number of public repos
                private_repos --- the number of private repos
                Contributor---previous contributor(CONTRIBUTOR or NONE)
                Organization Core Member---Is the author a project member?(Organization--->members_url)
            '''
            user, self.error_message = get_request_body(
                'https://api.github.com/users/' + pull_request['user']['login'] + '?access_token=' + self.token, self.token)
            user = user[0]
            if self.error_message is not None:
                return self.error_message

            if user['login'] in self.user_issues_merged.keys():
                issues_merged = self.user_issues_merged[user['login']]
            else:
                issues_merged, self.error_message = get_request_body(
                    'https://api.github.com/search/issues?' + 'q=type:pr+author:{}+is:unmerged+archived:false&sort=created&per_page=100&order=desc&access_token={}'.format(
                        user['login'], self.token), self.token)
                if self.error_message is not None:
                    return self.error_message

                tmp = []
                #issues_merged = issues_merged[0]
                for i in issues_merged:
                    tmp_dict = {}
                    tmp_dict['author_association'] = i['author_association']
                    tmp_dict['created_at'] = i['created_at']
                    tmp_dict['closed_at'] = i['closed_at']
                    tmp.append(tmp_dict)
                self.user_issues_merged[user['login']] = tmp
            if user['login'] in self.user_issues_all.keys():
                issues_all = self.user_issues_all[user['login']]
            else:
                issues_all, self.error_message = get_request_body(
                    'https://api.github.com/search/issues?' + 'q=type:pr+author:{}+archived:false&sort=created&order=desc&per_page=100&access_token={}'.format(
                        user['login'], self.token), self.token)
                #issues_all = issues_all[0]
                if self.error_message is not None:
                    return self.error_message
                tmp = []
                for i in issues_all:
                    tmp_dict = {}
                    tmp_dict['author_association'] = i['author_association']
                    tmp_dict['created_at'] = i['created_at']
                    tmp_dict['closed_at'] = i['closed_at']
                    tmp.append(tmp_dict)
                self.user_issues_all[user['login']] = tmp
            Prev_PRs = 0
            Accept_Num = 0
            Closed_Num = 0
            for i in issues_merged:
                if i['author_association'] != 'OWNER':
                    created_at_tmp_time = time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                    if created_at_tmp_time < end_time:
                        if i['closed_at'] is not None and time.strptime(i['closed_at'],
                                                                        "%Y-%m-%dT%H:%M:%SZ") < end_time:
                            Accept_Num += 1

            for i in issues_all:
                if i['author_association'] != 'OWNER':
                    created_at_tmp_time = time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                    if created_at_tmp_time < end_time:
                        Prev_PRs += 1
                        if i['closed_at'] is not None and time.strptime(i['closed_at'],
                                                                        "%Y-%m-%dT%H:%M:%SZ") < end_time:
                            Closed_Num += 1

            # Contributor
            CONTRIBUTOR = 0
            if pull_request['author_association'] == 'MEMBER':
                CONTRIBUTOR = 2
            elif pull_request['author_association'] == 'OWNER':
                CONTRIBUTOR = 3
            elif pull_request['author_association'] == 'COLLABORATOR':
                CONTRIBUTOR = 4
            else:
                CONTRIBUTOR = 1 if pull_request['author_association'] == 'CONTRIBUTOR' else 0
            # Organization or project core member or
            private_repos = 0
            public_repos = 0
            user_projects, self.error_message = get_request_body(user['repos_url'] + '?per_page=100&access_token=' + self.token, self.token) \
                if not (user['login'] in self.user_project.keys()) else self.user_project[user['login']]
            if self.error_message is not None:
                return self.error_message
            #user_projects = user_projects[0]
            for i in user_projects:
                tmp_time = time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                if tmp_time < end_time:
                    if i['private']:
                        private_repos += 1
                    else:
                        public_repos += 1
            organization_core_member = 0
            for i in self.members:
                if i['login'] == user['login']:
                    organization_core_member = 1
                    break
            tmp = {}
            tmp['Prev_PRs'] = Prev_PRs
            tmp['Followers'] = int(user['followers'])
            tmp['Following'] = int(user['following'])
            tmp['Accept_Num'] = Accept_Num
            tmp['Closed_Num'] = Closed_Num - Accept_Num
            tmp['User_Accept_Rate'] = Accept_Num / (Prev_PRs + 1e-10)
            tmp['Closed_Num_Rate'] = (Closed_Num - Accept_Num) / (Prev_PRs + 1e-10)
            tmp['Public_Repos'] = public_repos
            tmp['Private_Repos'] = private_repos
            tmp['Contributor'] = CONTRIBUTOR
            tmp['Organization_Core_Member'] = organization_core_member
            return tmp

        def project_features(pull_request):
            '''
                Project Features:
                projcet_age---age of project(Month)
                contributor_num---the number of contributor
                contributions---the number of contributions of the current user
                churn---total number of lines added and deleted by the pull request
                #Sloc---executable lines of code at creation time
                team_size---number of active core team members during the last 3 months prior to creation
                #perc_external_contribs---the ratio of commits from external members over core team members in the last 3 months prior to creation
                commits---Number of total commits  by the pull request 3 months before the creation time.
                file_touched ---Number of total files touched
                watchers---Project watchers (stars) at creation
                forks_count--- number of forked
                open_issues---the number of open issues
                accept_rate---the ratio of pr merged
                Merge_latency---the average time of merge of a pr
                Close_Latency
                language
                comments_per_merged_pr
                comments_per_closed_pr
                Statistics1---the last year of commit activity data(per week)
                Statistics2---number of additions and deletions per week
            '''
            Project_age = (time.mktime(end_time) - time.mktime(
                time.strptime(self.project[0]['created_at'], "%Y-%m-%dT%H:%M:%SZ"))) / 3600 / 30
            Contributions = 0
            for i in self.contributors:
                if 'login' in i.keys() and i['login'] == pull_request['user']['login']:
                    Contributions = i['contributions']
                    break
            Churn = 0
            Commits = 0
            File_Touched = 0
            pr_number = 0
            merge_num = 0
            pr_merge_latency = 0
            close_num = 0
            close_latency = 0
            merged_within_3_month = 0
            comments_per_closed_pr = 0
            comments_per_merged_pr = 0
            for i in self.pulls_details:
                # if time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ") < end_time :
                pr_number += 1
                if i['merged_at'] is not None:
                    Churn += i['additions']
                    Churn += i['deletions']
                    merge_num += 1
                    comments_per_merged_pr += i['comments']
                    pr_merge_latency += 0
                    Commits += i['commits']
                    File_Touched += i['changed_files']
                    merged_within_3_month += 1
                else:
                    close_num += 1
                    close_latency += 0
                    comments_per_closed_pr += i['comments']
            Forks_Count = 0
            for i in self.forks:
                if time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ") < end_time:
                    Forks_Count += 1
            Open_Issues = 0
            for i in self.issues:
                if not 'pull_request' in i.keys() and time.strptime(i['created_at'],
                                                                    "%Y-%m-%dT%H:%M:%SZ") < end_time and (
                        i['closed_at'] is None or time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ") > end_time):
                    Open_Issues += 1

            tmp = {}
            tmp['Project_Age'] = Project_age
            tmp['Contributions'] = Contributions
            tmp['Churn_Average'] = Churn / (merge_num + 1e-10)
            tmp['Open_Issues'] = Open_Issues
            tmp['Commits_Average'] = Commits / (merged_within_3_month + 1e-10)
            tmp['File_Touched_Average'] = File_Touched / (merged_within_3_month + 1e-10)
            tmp['Project_Accept_Rate'] = merge_num / (pr_number + 1e-10)
            tmp['Merge_Latency'] = pr_merge_latency / (merge_num + 1e-10)
            tmp['Close_Latency'] = close_latency / (close_num + 1e-10)
            tmp['Comments_Per_Merged_PR'] = comments_per_merged_pr / (merge_num + 1e-10)
            tmp['Comments_Per_Closed_PR'] = comments_per_closed_pr / (close_num + 1e-10)
            tmp['Forks_Count'] = Forks_Count
            tmp.update(self.Statistics)
            return tmp

        def label(pull_request):
            '''
                Label:Success---Is this pr merged successfully?
            '''
            tmp = {}
            if pull_request['closed_at'] is not None:
                if pull_request['merged_at'] is not None:
                    tmp['Label'] = 1
                else:
                    tmp['Label'] = 0
            return tmp

        def pull_request_features(pull_request):
            '''
                Pull Request Features:
                title---the title of pr
                body---the body of pr
                Intra-Branch---Are the source and target repositories the same?
                # Tested---Is this pr tested?
                Label_Count---the number of labels
                Additions---number of lines added
                Deletions---number of lines deleted
                Commits---number of commits
                Files_changed---number of files touched(sum of the above)
                workload---total number of pull requests still open in each project at current pull request creation time.
                mergeable_state---0:unstable 1:clean
                rebaseable---0:false 1:true
                mergeable---0:null 1:true 2:false
                Day---0:Sunday, 1:...
                Wait_Time
                Assignees_Count
                Requested_Reviewers_Count
                Requested_Teams_Count
            '''
            tmp = {}
            tmp['Title'] = pull_request['title']
            tmp['Body'] = pull_request['body_text'] if pull_request['body_text'] is not None else pull_request['body']
            tmp['Intra_Branch'] = 1 if pull_request['base']['ref'] == pull_request['head']['ref'] else 0
            tmp['Additions'] = pull_request['additions']
            tmp['Deletions'] = pull_request['deletions']
            tmp['Commits_PR'] = pull_request['commits']
            tmp['Files_Changed'] = pull_request['changed_files']
            Workload = 0
            for i in self.pulls_details:
                # if time.strptime(i['created_at'],"%Y-%m-%dT%H:%M:%SZ") < end_time and time.strptime(i['closed_at'],"%Y-%m-%dT%H:%M:%SZ") > end_time:
                Workload += 1
            tmp['Workload'] = Workload
            tmp['Mergeable_State'] = 1 if pull_request['mergeable_state'] is not None and pull_request[
                'mergeable_state'] == 'clean' else 0
            tmp['Rebaseable'] = 1 if pull_request['rebaseable'] is not None and pull_request['rebaseable'] else 0
            tmp['Mergeable'] = 1 if pull_request['mergeable'] is not None and pull_request['mergeable'] else 0
            tmp['Day'] = end_time[6]
            tmp['Wait_Time'] = (time.mktime(end_time) - time.mktime(
                time.strptime(pull_request['created_at'], "%Y-%m-%dT%H:%M:%SZ"))) / 3600 / 24
            tmp['Label_Count'] = len(pull_request['labels'])
            tmp['Assignees_Count'] = len(pull_request['assignees'])
            tmp['PR_Created_at'] = pull_request['created_at']
            return tmp

        def comment_features(pull_request):
            '''
                Comment Features:
                Comments Count---number of comment lines
                Review Comments Count---number of code review comments
                comments embedding(comments)
                review comments embedding(comments)
                num_commit_comments---the total number of code review comments
                Participants_Count---number of participants in the discussion
                Point_To_IssueOrPR
                Contains Fix---Is the pull request an issue fix?
                Last Comment Mention---Dose the last comment contain a user mention?
            '''
            comments, self.error_message = get_request_body(pull_request['comments_url'] + '?per_page=100&access_token=' + self.token, self.token)
            if self.error_message is not None:
                return self.error_message
            review_comments, self.error_message = get_request_body(
                pull_request['review_comments_url'] + '?per_page=100&access_token=' + self.token, self.token)
            if self.error_message is not None:
                return self.error_message
            Comments_Content = ''
            Review_Comments_Content = ''
            Fix_Bug = 0
            Point_To_IssueOrPR = 0
            if pull_request['title'] is not None and pull_request['title'].lower().find('fix') == 0 or pull_request[
                'body_text'] is not None and pull_request['body_text'].lower().find('fix') == 0:
                Fix_Bug = 1
            if pull_request['title'] is not None and pull_request['title'].lower().find('#') == 0 or pull_request[
                'body_text'] is not None and pull_request['body_text'].lower().find('#') == 0:
                Point_To_IssueOrPR = 1
            users = set()
            for i in comments:
                Comments_Content += i['body_text']
                users.add(i['user']['login'])
            for i in review_comments:
                Review_Comments_Content += i['body_text']
                users.add(i['user']['login'])
            tmp = {}
            tmp['Comments_Count'] = len(comments)
            tmp['Review_Comments_Count'] = len(review_comments)
            tmp['Comments_Embedding'] = Comments_Content
            tmp['Review_Comments_Embedding'] = Review_Comments_Content
            tmp['Participants_Count'] = len(users)
            if comments is not None and len(comments) != 0:
                tmp['Last_Comment_Mention'] = 1 if comments[-1]['body_text'].find('@') == 0 else 0
            else:
                tmp['Last_Comment_Mention'] = 0
            tmp['Contain_Fix_Bug'] = Fix_Bug
            tmp['Point_To_IssueOrPR'] = Point_To_IssueOrPR
            return tmp

        def response_features(pull_request):
            '''
                Date Features:
                Response_time---a list of key&value structs of response
            '''
            timeline = []
            body, self.error_message = get_request_body(
                'https://api.github.com/repos/{}/issues/{}/timeline?per_page=100'.format(self.project[0]['full_name'],
                                                                                         pull_request['number']), self.token,
                get_headers={'Accept': 'application/vnd.github.mockingbird-preview'})
            if self.error_message is not None:
                return self.error_message
            # print(body)
            for i in body:
                tmp_dict = {}
                if i['event'] == 'labeled':
                    tmp_dict['Type'] = 'labeled'
                    tmp_dict['Created_At'] = i['created_at']
                elif i['event'] == 'unlabeled':
                    tmp_dict['Type'] = 'unlabeled'
                    tmp_dict['Created_At'] = i['created_at']
                elif 'comments' in i.keys():
                    for j in i['comments']:
                        tmp_dict['Type'] = 'line-commented'
                        tmp_dict['Created_At'] = j['created_at']

                elif i['event'] == 'milestoned':
                    tmp_dict['Type'] = 'milestoned'
                    tmp_dict['Created_At'] = i['created_at']
                elif i['event'] == 'assigned':
                    tmp_dict['Type'] = 'assigned'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] == 'locked':
                    tmp_dict['Type'] = 'locked'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] == 'marked_as_duplicate':
                    tmp_dict['Type'] = 'marked_as_duplicate'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] == 'review_requested':
                    tmp_dict['Type'] = 'review_requested'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] == 'commented':
                    if i['author_association'] == 'MEMBER' or i['author_association'] == 'OWNER' or i[
                        'author_association'] == 'COLLABORATOR':
                        tmp_dict['Type'] = 'commented'
                        tmp_dict['Created_At'] = i['created_at']
                    else:
                        continue
                elif i['event'] == 'merged' or i['event'] == 'closed':
                    tmp_dict['Type'] = 'closed'
                    tmp_dict['Created_At'] = i['created_at']
                else:
                    continue
                timeline.append(tmp_dict)
            tmp = {}
            tmp['Timeline'] = timeline
            return tmp

        features = {}
        try:
            features.update(user_features(pull_request))
            features.update(project_features(pull_request))
            features.update(label(pull_request))
            features.update(pull_request_features(pull_request))
            features.update(comment_features(pull_request))
            features['url'] = pull_request['url']
            features.update(response_features(pull_request))
            # features.update(pull_request)
        except Exception as e:
            print(e)
            return None
        return features


def write_features_to_file(project_fullname, token, num_of_prs):
    dict_list = []
    print("PRs extraction in progress...")
    features = Features(token)
    error_message = features.check_links(project_fullname, token)

    if error_message is not None:
        return error_message
    for i in features.get_pull_request_features():
        if num_of_prs == 0:
            break
        if i is not None:
            dict_list.append(i)
            num_of_prs -= 1
    print('{} project extraction finished.'.format(project_fullname))

    return dict_list

# dict_list = write_features_to_file('PyGithub/PyGithub', '13c164006060562527a1eb9f644ef9718e6f3782', 5)

def file_is_empty(path):
    return os.stat(path).st_size==0


def extract_features(PR_list):
    print('Features extraction in progress')
    count = 0
    fieldnames = []
    fieldnames = list(PR_list[0].keys())
    fieldnames.append('X1_0')
    fieldnames.append('X1_1')
    fieldnames.append('X1_2')
    fieldnames.append('X1_3')
    fieldnames.append('X1_4')
    fieldnames.append('X1_5')
    fieldnames.append('X1_6')
    fieldnames.append('X1_7')
    fieldnames.append('X1_8')
    fieldnames.append('X1_9')
    fieldnames.append('X2_0')
    fieldnames.append('X2_1')
    fieldnames.append('X2_2')
    fieldnames.append('X2_3')
    fieldnames.append('X2_4')
    fieldnames.append('X2_5')
    fieldnames.append('X2_6')
    fieldnames.append('X2_7')
    fieldnames.append('X2_8')
    fieldnames.append('X2_9')
    fieldnames.append('PR_Latency')
    fieldnames.append('Pull_Request_ID')
    fieldnames.append('Project_Name')
    fieldnames.append('PR_Date_Created_At')
    fieldnames.append('PR_Time_Create_At')
    fieldnames.append('PR_Date_Closed_At')
    fieldnames.append('PR_Time_Closed_At')
    fieldnames.append('first_response')
    fieldnames.append('latency_after_first_response')
    fieldnames.append('wait_time_up')
    fieldnames.append('PR_response')
    fieldnames.append('PR_age')
    df = pd.DataFrame(columns=fieldnames)
    count=0
    for line in PR_list:
        try:
            #print(line)
            count = count + 1
            #print('Features extracted form PR: {}'.format(count))
            line['Pull_Request_ID'] = line['url'].split('/')[5]+'-'+line['url'].split('/')[7]
            line['Project_Name'] = line['url'].split('/')[5]
            current_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            line['PR_Date_Created_At'] = datetime.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ').date()
            line['PR_Time_Create_At'] = datetime.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ').time()
            line['PR_Date_Closed_At'] = 'NA'
            line['PR_Time_Closed_At'] = 'NA'
            # Calculate the first response of the pull reqeust

            check_response = 0
            if line['Timeline'] is not None:
                for item in range(1, len(line['Timeline'])):
                    type = str(line['Timeline'][item]['Type'])
                    if type == 'commented':
                        check_response += 1
                        # print(str(load_dict['Timeline'][item]['Created_At']))
                        line['first_response'] = (time.mktime(
                            time.strptime(str(line['Timeline'][item]['Created_At']),"%Y-%m-%dT%H:%M:%SZ"))
                                                       - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/60
                        line['latency_after_first_response'] = (time.mktime(
                            time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                                                     - time.mktime(
                                    time.strptime(str(line['Timeline'][item]['Created_At']), "%Y-%m-%dT%H:%M:%SZ"))) / 60
                        break

            if check_response == 0:
                line['first_response'] = (time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                               - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/60
                line['latency_after_first_response'] = (time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                                             - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/60
            line['PR_Latency'] = (time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                       - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/60
            line['wait_time_up'] = int((time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                             - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/3600/24)


            line['Wait_Time'] = (time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                      - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/3600/24


            line['PR_age'] = (time.mktime(time.strptime(str(current_date), "%Y-%m-%dT%H:%M:%SZ"))
                                   - time.mktime(time.strptime(str(line['PR_Created_at']), '%Y-%m-%dT%H:%M:%SZ')))/60

            if line['Title']:
                line['Title'] = line['Title'].replace("[\\p{P}+~$`^=|×]"," ")
            if line['Comments_Embedding']:
                line['Comments_Embedding'] = line['Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
            if line['Body']:
                line['Body'] = line['Body'].replace("[\\p{P}+~$`^=|<×]"," ")
            if line['Review_Comments_Embedding']:
                line['Review_Comments_Embedding'] = line['Review_Comments_Embedding'].replace("[\\p{P}+~$`^=|×]"," ")
            pattern = re.compile('^[a-zA-Z0-9]+$')
            size_TAB = 0
            size_CAR = 0
            list_Title=[0,0,0,0,0,0,0,0,0,0]
            if line['Title']:
                for item in line['Title'].split(" "):
                    if pattern.match(item) and model.wv.__contains__(item):
                        list_Title = [a+b for a,b in zip(model.wv.__getitem__(item),list_Title)]
            if line['Title']:
                size_TAB = size_TAB + len(line['Title'].split(" "))
            list_Comments_Embedding = [0,0,0,0,0,0,0,0,0,0]
            if line['Comments_Embedding']:
                for item in line['Comments_Embedding'].split(" "):
                    if pattern.match(item) and model.wv.__contains__(item):
                        list_Comments_Embedding = [a+b for a,b in zip(model.wv.__getitem__(item),list_Comments_Embedding)]
            if line['Comments_Embedding']:
                size_CAR = size_CAR + len(line['Comments_Embedding'].split(" "))
            list_Body=[0,0,0,0,0,0,0,0,0,0]
            if line['Body']:
                for item in line['Body'].split(" "):
                    if pattern.match(item) and model.wv.__contains__(item):
                        list_Body = [a+b for a,b in zip(model.wv.__getitem__(item),list_Body)]
            if line['Body']:
                size_TAB = size_TAB + len(line['Body'].split(" "))
            list_Review_Comments_Embedding=[0,0,0,0,0,0,0,0,0,0]
            if line['Review_Comments_Embedding']:
                for item in line['Review_Comments_Embedding'].split(" "):
                    if pattern.match(item) and model.wv.__contains__(item):
                        list_Review_Comments_Embedding = [a+b for a,b in zip(model.wv.__getitem__(item),list_Review_Comments_Embedding)]
            if line['Review_Comments_Embedding']:
                size_CAR = size_CAR + len(line['Review_Comments_Embedding'].split(" "))
            list_TAB = [a+b for a,b in zip(list_Title,list_Body)]
            for value in list_TAB:
                if value != 0:
                    value = value/size_TAB
            line['X1_0'] = list_TAB[0]
            line['X1_1'] = list_TAB[1]
            line['X1_2'] = list_TAB[2]
            line['X1_3'] = list_TAB[3]
            line['X1_4'] = list_TAB[4]
            line['X1_5'] = list_TAB[5]
            line['X1_6'] = list_TAB[6]
            line['X1_7'] = list_TAB[7]
            line['X1_8'] = list_TAB[8]
            line['X1_9'] = list_TAB[9]
            list_CAR = [a+b for a,b in zip(list_Comments_Embedding,list_Review_Comments_Embedding)]
            for value in list_CAR:
                if value != 0:
                    value = value/size_CAR
            line['X2_0'] = list_CAR[0]
            line['X2_1'] = list_CAR[1]
            line['X2_2'] = list_CAR[2]
            line['X2_3'] = list_CAR[3]
            line['X2_4'] = list_CAR[4]
            line['X2_5'] = list_CAR[5]
            line['X2_6'] = list_CAR[6]
            line['X2_7'] = list_CAR[7]
            line['X2_8'] = list_CAR[8]
            line['X2_9'] = list_CAR[9]
            df = df.append(line, ignore_index=True)
        except Exception as e:
            print(e)
            continue
    return df




def encode_labels(df1, column_name):
    encoder = preprocessing.LabelEncoder()
    df1[column_name] = [str(label) for label in df1[column_name]]
    encoder.fit(df1[column_name])
    one_hot_vector = encoder.transform(df1[column_name])
    return  one_hot_vector


def Cartesian_models(df_r):
    pd.options.mode.chained_assignment = None
    predictors_a = ['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size',
                    'Additions_Per_Week', 'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Churn_Average', 'Close_Latency',
                    'Comments_Per_Closed_PR', 'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Additions', 'Deletions',
                    'Wait_Time', 'PR_Latency', 'Files_Changed', 'Label_Count', 'Workload', 'Commits_Average',
                    'Contributor', 'Followers', 'Closed_Num', 'Public_Repos', 'Accept_Num', 'User_Accept_Rate', 'Contributions',
                    'Closed_Num_Rate', 'Prev_PRs', 'Open_Issues', 'first_response', 'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2',
                    'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9']
    predictors_r = ['Project_Age', 'Project_Accept_Rate', 'Language', 'Watchers', 'Stars', 'Team_Size',
                    'Additions_Per_Week', 'Deletions_Per_Week', 'Comments_Per_Merged_PR', 'Contributor_Num', 'Churn_Average', 'Sunday',
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Close_Latency', 'Comments_Per_Closed_PR',
                    'Forks_Count', 'File_Touched_Average', 'Merge_Latency', 'Rebaseable', 'Intra_Branch',
                    'Additions', 'Deletions', 'Day', 'Commits_PR', 'Wait_Time', 'Contain_Fix_Bug', 'PR_Latency',
                    'Files_Changed', 'Label_Count', 'Assignees_Count', 'Workload', 'PR_age', 'Commits_Average', 'Contributor',
                    'Followers', 'Closed_Num', 'Public_Repos', 'Organization_Core_Member', 'Accept_Num', 'User_Accept_Rate',
                    'Contributions', 'Closed_Num_Rate', 'Following', 'Prev_PRs', 'Review_Comments_Count', 'Participants_Count',
                    'Comments_Count', 'Last_Comment_Mention', 'Point_To_IssueOrPR', 'Open_Issues', 'first_response',
                    'latency_after_first_response', 'X1_0', 'X1_1', 'X1_2', 'X1_3', 'X1_4', 'X1_5', 'X1_6', 'X1_7', 'X1_8', 'X1_9']

    for col in df_r[predictors_a].columns:
        if df_r[col].dtypes=='object':
            df_r[col] = encode_labels(df_r, col)

    for col in df_r[predictors_r].columns:
        if df_r[col].dtypes=='object':
            df_r[col] = encode_labels(df_r, col)


    with open('Models/accept_XGB_tool.pickle.dat', 'rb') as f:
        accept_model = pickle.load(f)
        y_pred_accept = accept_model.predict_proba(df_r[predictors_a])[:,1]
        df_r['Result_Accept'] = y_pred_accept

    with open('Models/response_XGB_tool.pickle.dat', 'rb') as f:
        response_model = pickle.load(f)
        y_pred_response = response_model.predict_proba(df_r[predictors_r])[:,1]
        df_r['Result_Response'] = y_pred_response

    df_r['Score'] = df_r['Result_Accept'].apply(np.exp) + df_r['Result_Response'].apply(np.exp)

    df_r = df_r.sort_values(by=['Score'], ascending=False)

    return df_r[['Pull_Request_ID', 'Title']]
