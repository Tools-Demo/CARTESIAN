# -*- coding: utf-8 -*-
import itertools
import json
import os
import logging
import requests
import time
logging.getLogger().setLevel(logging.INFO)

class OrganizationCrawler(object):
	def __init__(self):
		self.start_url = 'https://api.github.com/organizations'
		self.token_list=[
			'eb532c1ee094c06aa1f9279812cfc37fdb919f3f',
			'049f396818e2258a9c21903d37c8d6aaccb15bb4'
		]
		self.output_file = open('../data/organization.txt', "a+")
		self.token_iter = itertools.cycle(self.token_list)
	def crawler(self):
		loop_url = self.start_url + '?since=19337918&access_token='+ next(self.token_iter)
		while True:
			try:
				response = requests.get(loop_url)
				logging.info('crawler url:{}, status:{}, remaining times:{}'.format(loop_url, response.status_code, response.headers['X-RateLimit-Remaining']))
				if response.status_code == requests.codes.ok:
					data = response.json()
					for tmp in data:
						self.output_file.write(json.dumps(tmp)+'\n')
					if 'next' in response.links.keys():
						loop_url= response.links['next']['url'] +'&access_token=' + next(self.token_iter)
					else:
						logging.info('------------------------------end-------------------------')
						self.output_file.close()
						break
					if int(response.headers['X-RateLimit-Remaining']) == 0:
						logging.info('------------------------------sleep {} seconds.-------------------------'.format(int(response.headers['X-RateLimit-Reset'])/1000+10))
						time.sleep(int(response.headers['X-RateLimit-Reset'])/1000+10) 
				else:
					logging.info('------------------------------sleep {} seconds.-------------------------'.format(int(response.headers['X-RateLimit-Reset'])/1000+10))
					time.sleep(int(response.headers['X-RateLimit-Reset'])/1000+10)
			except Exception as e:
				logging.info(e)
				time.sleep(60)


if __name__ == '__main__':
	OrganizationCrawler().crawler()

