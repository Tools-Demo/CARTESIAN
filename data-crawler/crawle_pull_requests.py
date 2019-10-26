#!/usr/bin/env python
#-*-coding:utf-8-*-

'''
    Author: pq
    Function: Features Extractor

'''
import logging
import json
import requests
import itertools
import time
import threading
import sys
# log_file = open('log.txt','w')
# stdo=sys.stdout
# ster=sys.stderr
# sys.stderr=log_file
# sys.stdout=log_file
# logging.getLogger().setLevel(logging.INFO)
# error_log = open('error_log.txt','w')
input_file = './project-part/part-2.txt'
output_file = './raw-data/raw-data-part-0.txt'
token_list=[
                'eb532c1ee094c06aa1f9279812cfc37fdb919f3f',
			    '049f396818e2258a9c21903d37c8d6aaccb15bb4'
            ]
number = 5000
token_iter = itertools.cycle(token_list)
less_con = 0
def flat(tree):
    res = []
    for i in tree:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def get_request_body(url,get_headers = None):
    body=[]
    loop_url = url+'&access_token='+next(token_iter)
    #print(url)
    headers = {'Accept':'application/vnd.github.v3.text+json'} if get_headers is None else get_headers
    #print(headers)
    while True:
                try:
                    response = requests.get(loop_url, timeout=10,headers=headers)
                    if response.status_code == requests.codes.ok:
                        data = response.json()
                        if isinstance(data,dict) and 'items' in data.keys():
                            body.append(data['items'])
                        elif isinstance(data,dict) and 'errors' in data.keys():
                            print('Arguments Error: {}'.format(loop_url))
                            # error_log.write('Arguments Error:'+loop_url+'\n')
                            # error_log.flush()
                        else:
                            body.append(data)
                        if 'next' in response.links.keys():
                            loop_url = response.links['next']['url']
                        else:
                            break
                        if int(response.headers['X-RateLimit-Remaining']) == 0:
                            logging.warning('Sleep: {}s because of rateLimit'.format(600))
                            time.sleep(600)
                    elif str(response.status_code) == '404' or str(response.status_code)=='451':
                        print('Status Error: {}'.format(loop_url))
                        # error_log.write('Status Error:'+loop_url+'\n')
                        # error_log.flush()
                        break
                    elif str(response.status_code) =='403':
                        print('Status: {}, Sleep: {}s '.format(response.status_code, 60))
                        print('Status Error: {}'.format(loop_url))
                        # logging.warning('Status: {}, Sleep: {}s '.format(response.status_code, 60))
                        # error_log.write('Status Error:'+loop_url+'\n')
                        # error_log.flush()
                        time.sleep(60)
                    else:
                        print('Status: {}, Sleep: {}s '.format(response.status_code, 60))
                        print('Status Error: {}'.format(loop_url))
                        # logging.warning('Status：{}, Sleep: {}s'.format(response.status_code, 60))
                        # error_log.write('Status Error:'+loop_url+'\n')
                        # error_log.flush()
                        time.sleep(60)
                except Exception as e:
                    print('$$$$$$$$ Exception: {}, Sleep: {}s'.format(e, 60))
                    # logging.warning('$$$$$$$$ Exception: {}, Sleep: {}s'.format(e, 60))
                    # time.sleep(60)
    return flat(body)

class Features:
    def __init__(self,project,token):
        self.token = token
        self.project = get_request_body('https://api.github.com/repos/'+project['full_name']+'?per_page=100&access_token='+self.token)[0]

        print('Project API called')

        self.contributors = get_request_body(self.project['contributors_url']+'?anon=1&per_page=100&access_token='+self.token)

        print('Contributors API called')

        if len(self.contributors) < 1000:
            return
        pulls = get_request_body('https://api.github.com/repos/' + self.project['full_name'] + '/pulls?state=open&per_page=20&access_token='+self.token)

        print('Pull Request API called')

        #self.pulls_details = []
        #for i in self.pulls:
        #    self.pulls_details.append(get_request_body(i['url']+'?access_token='+self.token)[0])
        self.issues = []
        for i in get_request_body('https://api.github.com/repos/{}/issues?per_page=100&access_token={}'.format(self.project['full_name'],self.token)):
            tmp_dict = {}
            tmp_dict['created_at'] = i['created_at']
            tmp_dict['closed_at'] = i['closed_at']
            if 'pull_request' in i.keys():
                tmp_dict['pull_request'] = i['pull_request']
            self.issues.append(tmp_dict)
        self.members = get_request_body('https://api.github.com/orgs/{}/members?per_page=100&access_token={}'.format(self.project['organization']['login'],self.token))
        Statistics1 = get_request_body('https://api.github.com/repos/'+self.project['full_name']+'/stats/commit_activity?per_page=100&access_token='+self.token)
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
        Statistics2 = get_request_body('https://api.github.com/repos/'+self.project['full_name']+'/stats/code_frequency?per_page=100&access_token='+self.token)
        Additions_Per_Week = 0
        Deletions_Per_Week = 0
        for i in range(int(len(Statistics2)/3)):
            Additions_Per_Week += Statistics2[i*3+1]
            Deletions_Per_Week += Statistics2[i*3+2]
        self.Statistics = {}
        self.Statistics['Sunday'] = Sunday/ (len(Statistics1)+1e-10)
        self.Statistics['Monday'] = Monday/ (len(Statistics1)+1e-10)
        self.Statistics['Tuesday'] = Tuesday/ (len(Statistics1)+1e-10)
        self.Statistics['Wednesday'] = Wednesday/ (len(Statistics1)+1e-10)
        self.Statistics['Thursday'] = Thursday/ (len(Statistics1)+1e-10)
        self.Statistics['Friday'] = Friday/ (len(Statistics1)+1e-10)
        self.Statistics['Saturday'] = Saturday/ (len(Statistics1)+1e-10)
        self.Statistics['Additions_Per_Week'] = Additions_Per_Week / (len(Statistics2)/3+1e-10)
        self.Statistics['Deletions_Per_Week'] = Deletions_Per_Week / (len(Statistics2)/3+1e-10)
        self.Statistics['Contributor_Num'] = len(self.contributors)
        self.Statistics['Language'] = self.project['language']
        self.Statistics['Team_Size'] = len(self.members)
        self.Statistics['Watchers'] = self.project['subscribers_count']
        self.Statistics['Stars'] = self.project['watchers']
        self.pulls_details = []
        for i in pulls:
            tmp = get_request_body(i['url']+'?access_token='+self.token)[0]
            tmp_dict={}
            tmp_dict['user'] = {'type':tmp['user']['type'], 'login':tmp['user']['login'], 'repos_url':tmp['user']['repos_url']}
            tmp_dict['closed_at'] = tmp['closed_at']
            tmp_dict['author_association'] = tmp['author_association']
            tmp_dict['created_at'] = tmp['created_at']
            tmp_dict['merged_at'] = tmp['merged_at']
            tmp_dict['deletions'] = tmp['deletions']
            tmp_dict['comments'] = tmp['comments']
            tmp_dict['changed_files'] = tmp['changed_files']
            tmp_dict['commits'] = tmp['commits']
            tmp_dict['title'] = tmp['title'] if tmp['title'] is not None else None
            tmp_dict['body'] = tmp['body']
            tmp_dict['base'] = {'ref':tmp['base']['ref']}
            tmp_dict['head'] = {'ref':tmp['head']['ref']}
            tmp_dict['mergeable_state'] = tmp['mergeable_state']
            tmp_dict['rebaseable'] = tmp['rebaseable']
            tmp_dict['mergeable'] = tmp['mergeable']
            tmp_dict['labels'] = tmp['labels']
            tmp_dict['body_text'] = tmp['body_text']
            tmp_dict['review_comments_url'] = tmp['review_comments_url']
            tmp_dict['comments_url'] = tmp['comments_url']
            tmp_dict['assignees'] = tmp['assignees']
            tmp_dict['additions'] = tmp['additions']
            tmp_dict['url'] = tmp['url']
            tmp_dict['number'] = tmp['number']
            self.pulls_details.append(tmp_dict)
        self.user_issues_all = {}
        self.user_issues_merged = {}
        self.forks = []
        for i in get_request_body('https://api.github.com/repos/' + self.project['full_name'] + '/forks?per_page=100&access_token='+self.token):
            self.forks.append({'created_at':i['created_at']})

        print('Forks API called')

        self.user_project = {}
    def get_pull_request_features(self):
        if len(self.contributors) < 1000:
            return
        for i in self.pulls_details:
            if i['user']['type'] == 'User':
                yield self.getFeatures(i)

    def getFeatures(self, pull_request):
        # end_time = time.strptime(pull_request['closed_at'],"%Y-%m-%dT%H:%M:%SZ")
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
            user = get_request_body('https://api.github.com/users/'+pull_request['user']['login']+'?access_token='+self.token)[0]

            print('Users API called')

            if user['login'] in self.user_issues_merged.keys():
                issues_merged = self.user_issues_merged[user['login']]
            else:
                issues_merged = get_request_body('https://api.github.com/search/issues?' + 'q=type:pr+author:{}+is:unmerged+archived:false&sort=created&per_page=100&order=desc&access_token={}'.format(user['login'],self.token))

                print('Issues_merged API called')

                tmp = []
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
                issues_all = get_request_body('https://api.github.com/search/issues?' + 'q=type:pr+author:{}+archived:false&sort=created&order=desc&per_page=100&access_token={}'.format(user['login'],self.token))
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
            # for i in issues_merged:
            #     if i['author_association'] !='OWNER':
            #         created_at_tmp_time = time.strptime(i['created_at'],"%Y-%m-%dT%H:%M:%SZ")
            #         if created_at_tmp_time < end_time:
            #             if i['closed_at'] is not None and time.strptime(i['closed_at'],"%Y-%m-%dT%H:%M:%SZ") < end_time:
            #                 Accept_Num +=1

            # for i in issues_all:
            #     if i['author_association'] !='OWNER':
            #         created_at_tmp_time = time.strptime(i['created_at'],"%Y-%m-%dT%H:%M:%SZ")
            #         if created_at_tmp_time < end_time:
            #             Prev_PRs += 1
            #             if i['closed_at'] is not None and time.strptime(i['closed_at'],"%Y-%m-%dT%H:%M:%SZ")<end_time:
            #                 Closed_Num +=1

            # Contributor
            CONTRIBUTOR = 0
            if pull_request['author_association'] =='MEMBER':
                CONTRIBUTOR = 2
            elif pull_request['author_association'] == 'OWNER':
                CONTRIBUTOR = 3
            elif pull_request['author_association'] == 'COLLABORATOR':
                CONTRIBUTOR = 4
            else:
                CONTRIBUTOR = 1 if pull_request['author_association']=='CONTRIBUTOR' else 0
            # Organization or project core member or 
            private_repos = 0
            public_repos = 0
            user_projects = get_request_body(user['repos_url']+'?per_page=100&access_token='+self.token) \
                if not (user['login'] in self.user_project.keys()) else self.user_project[user['login']]
            for i in user_projects:
                # tmp_time = time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                # if tmp_time < end_time:
                if i['private']:
                    private_repos +=1
                else:
                    public_repos +=1
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
            tmp['User_Accept_Rate'] = Accept_Num / (Prev_PRs+1e-10)
            tmp['Closed_Num_Rate'] = (Closed_Num-Accept_Num) / (Prev_PRs+1e-10)
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
            Project_age = 0 #(time.mktime(end_time) - time.mktime(time.strptime(self.project['created_at'], "%Y-%m-%dT%H:%M:%SZ")))/3600/30
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
            comments_per_closed_pr=0
            comments_per_merged_pr=0
            for i in self.pulls_details:
                # if time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ") < end_time :
                    pr_number +=1
                    if i['merged_at'] is not None:
                        Churn += i['additions']
                        Churn += i['deletions']
                        merge_num +=1
                        comments_per_merged_pr +=i['comments']
                        # pr_merge_latency += (time.mktime(time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")))/3600/24
                        # if time.mktime(end_time) - time.mktime(time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ")) >=0 and time.mktime(end_time) - time.mktime(time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ")) <= 90*24*3600:
                        Commits += i['commits']
                        File_Touched += i['changed_files']
                        merged_within_3_month +=1
                    else:
                        close_num +=1
                        # close_latency += (time.mktime(time.strptime(i['closed_at'],"%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ")))/3600/24
                        comments_per_closed_pr += i['comments']
            Forks_Count = 0
            for i in self.forks:
                # if time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ") < end_time:
                Forks_Count +=1
            Open_Issues = 0
            for i in self.issues:
                if not 'pull_request' in i.keys(): #and  time.strptime(i['created_at'], "%Y-%m-%dT%H:%M:%SZ") < end_time and (i['closed_at'] is None or time.strptime(i['closed_at'], "%Y-%m-%dT%H:%M:%SZ")>end_time):
                    Open_Issues +=1

            tmp = {}
            tmp['Project_Age'] = Project_age
            tmp['Contributions'] = Contributions
            tmp['Churn_Average'] = Churn/ (merge_num+1e-10)
            tmp['Open_Issues'] = Open_Issues
            tmp['Commits_Average'] = Commits / (merged_within_3_month+1e-10)
            tmp['File_Touched_Average'] = File_Touched / (merged_within_3_month+1e-10)
            tmp['Project_Accept_Rate'] = merge_num / (pr_number+1e-10)
            tmp['Merge_Latency'] = pr_merge_latency / (merge_num+1e-10)
            tmp['Close_Latency'] = close_latency / (close_num+1e-10)
            tmp['Comments_Per_Merged_PR'] = comments_per_merged_pr / (merge_num +1e-10)
            tmp['Comments_Per_Closed_PR'] = comments_per_closed_pr / (close_num +1e-10)
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
                #if time.strptime(i['created_at'],"%Y-%m-%dT%H:%M:%SZ") < end_time and time.strptime(i['closed_at'],"%Y-%m-%dT%H:%M:%SZ") > end_time:
                Workload+=1
            tmp['Workload'] = Workload
            tmp['Mergeable_State'] = 1 if pull_request['mergeable_state'] is not None and  pull_request['mergeable_state'] =='clean' else 0
            tmp['Rebaseable'] = 1 if pull_request['rebaseable'] is not None and pull_request['rebaseable']  else 0
            tmp['Mergeable'] = 1 if pull_request['mergeable'] is not None and pull_request['mergeable']  else 0
            # tmp['Day'] = end_time[6]
            tmp['Day'] = time.strptime(pull_request['created_at'], "%Y-%m-%dT%H:%M:%SZ")[6]
            # tmp['Wait_Time'] = (time.mktime(end_time) - time.mktime(time.strptime(pull_request['created_at'], "%Y-%m-%dT%H:%M:%SZ")))/3600/24
            tmp['Wait_Time'] = 0
            tmp['Label_Count'] = len(pull_request['labels'])
            tmp['Assignees_Count'] = len(pull_request['assignees'])
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
            comments = get_request_body(pull_request['comments_url']+'?per_page=100&access_token='+self.token)
            review_comments = get_request_body(pull_request['review_comments_url']+'?per_page=100&access_token='+self.token)
            Comments_Content = ''
            Review_Comments_Content =''
            Fix_Bug = 0
            Point_To_IssueOrPR = 0
            if pull_request['title'] is not None and pull_request['title'].lower().find('fix') == 0 or pull_request['body_text'] is not None and pull_request['body_text'].lower().find('fix') == 0:
                Fix_Bug = 1
            if pull_request['title'] is not None and pull_request['title'].lower().find('#') == 0 or pull_request['body_text'] is not None and pull_request['body_text'].lower().find('#') == 0:
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
                tmp['Last_Comment_Mention'] = 1 if comments[-1]['body_text'].find('@')==0 else 0
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
            body = get_request_body('https://api.github.com/repos/{}/issues/{}/timeline?per_page=100'.format(self.project['full_name'], pull_request['number']),get_headers={'Accept':'application/vnd.github.mockingbird-preview'})
            #print(body)
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

                elif i['event'] =='marked_as_duplicate':
                    tmp_dict['Type'] = 'marked_as_duplicate'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] =='review_requested':
                    tmp_dict['Type'] = 'review_requested'
                    tmp_dict['Created_At'] = i['created_at']

                elif i['event'] =='commented':
                    if i['author_association'] == 'MEMBER' or i['author_association'] == 'OWNER' or i['author_association'] == 'COLLABORATOR':
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
            features['url']=pull_request['url']
            features.update(response_features(pull_request))
            #features.update(pull_request)
        except Exception as e:
            print(e)
            return None
        return features

def write_features_to_file(name, project, token_iter,file_writer):
    print('write_features_to_file method called')
    print('Thread:{} Start init.'.format(name))
    features = Features(project, token_iter)
    print('Thread:{} end init.'.format(name))
    for i in features.get_pull_request_features():
        if i is not None:
            print('Project: {}, examples: {}'.format(name))
            file_writer.write(json.dumps(i)+'\n')
    print('{} finished.'.format(name))

if __name__ == '__main__':
    file_writer = open(output_file, 'a+')
    print("Data extraction in progress...")
    threads_list=[]
    with open(input_file, 'r') as f:
        for line in f.readlines():
            project = json.loads(line.strip())
            print(project['full_name'])
            if int(project['pull_request_number']) >=number:
                write_features_to_file(project['full_name'], project, next(token_iter), file_writer)
                # threads_list.append(threading.Thread(target=write_features_to_file, args=(project['full_name'], project, next(token_iter), file_writer)))

    # print('threads_list length: {}'.format(len(threads_list)))
    # for i in threads_list:
    #     i.start()
    #     start = 0
    #     end = 8
    #     while True:
    #         if threading.activeCount()<=1 and end >= len(threads_list):
    #             break
    #         print('活着的线程数量为{}'.format(threading.activeCount()))
    #         if start == end and end == len(threads_list):
    #             print('end~')
    #             break
    #         tmp_end = end
    #         tmp_start = start
    #         for i in range(tmp_start,tmp_end):
    #             if not threads_list[i].is_alive():
    #                 print('Thread end:{}'.format(i))
    #                 if end < len(threads_list):
    #                     threads_list[end].start()
    #                     end +=1
    #                 print('Thread start:{}'.format(i))
    #                 if i == start:
    #                     start+=1
    #         tmp = threading.activeCount()
    #         for i in range(9 - tmp):
    #             if end < len(threads_list):
    #                 threads_list[end].start()
    #                 print('Thread: {} Start.'.format(end))
    #                 end +=1
    #         time.sleep(3600)
    # for i in threads_list:
    #     i.join()
    print('Program ends')
    # sys.stdout=stdo
    # sys.stderr=ster
