# -*- coding:utf-8 -*-
import itertools
import json
import logging
import requests
import time
import threading
#创建锁
mutex = threading.Lock()
lines=0
#锁定
#mutex.acquire([timeout])
#释放
#mutex.release()
file_organization = open('../data/organization.txt', 'r')
file_project = open('../data/project.txt', 'a+')
file_log = open('log_url.txt','a+')
logging.getLogger().setLevel(logging.INFO)
token_list=[
			'eb532c1ee094c06aa1f9279812cfc37fdb919f3f',
			'049f396818e2258a9c21903d37c8d6aaccb15bb4'
		]
def generateProjects(file, token, identification):
	# Read File
	global lines
	name = identification
	while True:
		if mutex.acquire(True):
			line = file.readline()
			lines +=1
			logging.info('Lines: {}, From: {}'.format(lines, name))
			mutex.release()
		if line == '': 
			logging.info(' ------------------------  done. From: {} ---------------------------'.format(name))
			break
		organization = json.loads(line.strip())
		loop_url = organization['repos_url']+'?&access_token='+token
		while True:
			try:
				file_log.write('From: {}, loop_url: {} \n'.format(name, loop_url))
				response = requests.get(loop_url)
				if response.status_code == requests.codes.ok:
					data = response.json()
					for tmp in data:
						tmp_data = {}
						tmp_data['id'] = tmp['id']
						tmp_data['full_name'] = tmp['full_name']
						tmp_data['name'] = tmp['name']
						tmp_data['pulls_url'] = tmp['pulls_url']
						tmp_data['forks_count'] = tmp['forks_count']
						tmp_data['size'] = tmp['size']
						file_project.write(json.dumps(tmp_data)+'\n')
					if 'next' in response.links.keys(): 
						loop_url = response.links['next']['url']
					else:
						break
					if int(response.headers['X-RateLimit-Remaining']) == 0:
						logging.warning('Sleep: {}s, From: {}.'.format(600, name))
						time.sleep(600)
				elif str(response.status_code) == '404':
					break
				else:
					logging.warning('Status: {}, From: {}, Sleep: {}s, '.format(response.status_code, name, 600))
					time.sleep(600)
			except Exception as e:
				logging.warning('$$$$$$$$ From: {}, Exception: {}, Sleep {}s'.format(name, e, 60))
				time.sleep(60)


for i in range(len(token_list)):
	t = threading.Thread(target = generateProjects, args=(file_organization, token_list[i], i,))
	t.start()


logging.info('-------------------------finished---------------------------')