from flask import Flask, render_template, request, redirect, session, url_for
from flask_cors import CORS
from flask import jsonify
import PRs_Extractor as prEx
from flask import send_from_directory
# from werkzeug.utils import secure_filename
import logging
import codecs
import datetime
import json
import uuid
import re
import os
# import cart

#app = Flask(__name__)
app = Flask(__name__,static_folder='', static_url_path='')
CORS(app)  # 解决跨域
app.debug = True  # 自动重启
SECRET_KEY = '*\xff\x93\xc8w\x13\x0e@3\xd6\x82\x0f\x84\x18\xe7\xd9\\|\x04e\xb9(\xfd\xc3'
app.config['SECRET_KEY'] = 'SECRET_KEY'
list_dict = None
list_features = None
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/pull', methods=['GET', 'POST'], endpoint='l1')  # endpoint 表示别名
def pull():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.get_json("data")
        repo = data['repo']
        token = data['token']
        number = data['num']
        try:
            number = int(number)
        except ValueError:
            number = -1
        if repo or token or number != -1:
            logging.info('Pull request extraction in progress..')
            # prEx.access_token= access_token
            global list_dict
            list_dict = prEx.write_features_to_file(repo, token, number)
            if type(list_dict) == str:
                logging.info('Pull request extraction failed')
                return jsonify({"success": 200, "msg": list_dict, "tag": "wrong"})
            logging.info('Pull request extraction completed')
            return jsonify({"success": 200, "msg": "Pull request extraction completed!", "tag": "true"})
        return jsonify({"success": 403, "msg": "You\'ve input wrong format data!"})


@app.route('/features', methods=['GET', 'POST'], endpoint='l2')  # endpoint 表示别名
def features():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.get_json("data")
        repo = data['repo']
        token = data['token']
        number = data['num']
        try:
            number = int(number)
        except ValueError:
            number = -1
        if repo or token or number != -1:
            logging.info('Features extraction in progress..')
            global list_dict
            global list_features
            if type(list_dict) == 'NoneType':
                return jsonify({"success": 200, "msg": "The features list is empty!", "tag": "wrong"})
            list_features = prEx.extract_features(list_dict)
            logging.info('Features extraction completed')
            return jsonify({"success": 200, "msg": "Features extraction completed!!", "tag": "true"})
        return jsonify({"success": 403, "msg": "You\'ve input wrong format data!"})\


@app.route('/model', methods=['GET', 'POST'], endpoint='l3')  # endpoint 表示别名
def model():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.get_json("data")
        repo = data['repo']
        token = data['token']
        number = data['num']
        try:
            number = int(number)
        except ValueError:
            number = -1
        if repo or token or number != -1:
            global list_features
            if type(list_features) == 'NoneType':
                return jsonify({"success": 200, "msg": "Features list is empty", "tag": "wrong"})

            list_PRs = prEx.Cartesian_models(list_features)

            logging.info('Processing completed...')
            tablex = '<tr bgcolor="##0080FF"><th>ID</th><th>Pull Request Title</th><th></th></tr>'
            for index, row in list_PRs.iterrows():
                idx = str(row[0])
                titlex = str(row[1])
                tablex += '<tr><td>'+idx+'</td><td>'+titlex+'</td><td><input name="like_'+idx+'" type="radio" value="like" />like</label><input name="like_'+idx+'" type="radio" value="dislike" />dislike</label></td></tr>'
            # tablex += '</table>'
            return jsonify({"success": 200, "msg": "Processing completed!", "table": tablex, "tag": "true"})
        return jsonify({"success": 403, "msg": "You\'ve input wrong format data!"})

@app.route('/file_store', methods=['GET', 'POST'], endpoint='l4')  # endpoint 表示别名
def file_store():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = request.get_json("data")
        stog = data['pref']
        result = []
        for line in stog:
            result.append({"uuid": str(uuid.uuid1()) , "ID": line[0], "Pull_Request_Title": line[1] , "prefer": line[2] , "time": str(datetime.datetime.now())})
        with open('C:/Users/Administrator/Desktop/CARTESIAN/Json_file/storage.json', 'a+') as f:
            json.dump(result, f)
        return jsonify({"success": 200, "msg": "succeed!"})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7000)
