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
logging.getLogger().setLevel(logging.INFO)
input_file = '../data/project_pull_request_number_closed.txt.2018-01-27'
output_file = '../data-generate/selected_project.txt'
token_list=[
                    'be77d477afb39c6345e649d0288b5c83ee024c4f',
                    'd0ee97a857d1e8e4c4918593bacfd69eaf011612',
                    '4cd06100b2c2234a3f02151e4f1ee1570c5b8a19',
                    'd4b89ddd287e96b9a68774502ab6809521229db7',
                    'e21995fb5f942669b2a3edd2b3b754e568e33217',
                    '3a867c56d662d7bd9778d35b653a7b19790b6777',
                    'fe5583b93cbcd0fc02d751530cfc5d489804524b',
                    'e5bb25ef369589e90e200cb1a528f1820bdb9994',
                    '6cef8e2c458c84b776d82c3e0b29bde46fb6d2cf',
                    '2719445c05991f1e5febfef1a09ca8befd042ddd',
                    'dcefdee459ea41ffd80b0372f056ca1a7aec49a1',
                    '029e57649d47dda6f0d81d6643518ba8af153626',
                    'c19f2995cdbe35ffe335d9e94573c8b75cb9bce1'
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

def get_request_body(url):
    body=[]
    loop_url = url+'&access_token='+next(token_iter)
    #print(url)
    headers = {'Accept':'application/vnd.github.v3.text+json'}
    while True:
                try:
                    response = requests.get(loop_url, timeout=10,headers=headers)
                    if response.status_code == requests.codes.ok:
                        data = response.json()
                        if isinstance(data,dict) and 'items' in data.keys():
                            body.append(data['items'])
                        elif isinstance(data,dict) and 'errors' in data.keys():
                            error_log.write('Arguments Error:'+loop_url+'\n')
                            error_log.flush()
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
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        break
                    elif str(response.status_code) =='403':
                        logging.warning('Status: {}, Sleep: {}s '.format(response.status_code, 60))
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        time.sleep(60)
                    else:
                        logging.warning('Status：{}, Sleep: {}s'.format(response.status_code, 60))
                        error_log.write('Status Error:'+loop_url+'\n')
                        error_log.flush()
                        time.sleep(60)
                except Exception as e:
                    logging.warning('$$$$$$$$ Exception: {}, Sleep: {}s'.format(e, 60))
                    time.sleep(60)
    return flat(body)

class Features:
    def __init__(self,project,token):
        self.token = token
        self.project = get_request_body('https://api.github.com/repos/'+project['full_name']+'?per_page=100&access_token='+self.token)[0]
        self.contributors = get_request_body(self.project['contributors_url']+'?anon=1&per_page=100&access_token='+self.token)

        
if __name__ == '__main__':
    with open(input_file, 'r') as f, open(output_file, 'w') as w :
        for line in f.readlines():
            project = json.loads(line.strip())
            if int(project['pull_request_number']) >= number:
                tmp = Features(project, next(token_iter))
                if len(tmp.contributors) >= 1000:
                    w.write(json.dumps(project)+'\n')