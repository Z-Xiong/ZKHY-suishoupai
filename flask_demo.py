from flask import Flask, render_template, Response,jsonify,request
import torch
from torchvision import transforms as tfs
from torch import nn
import torchvision.models as models
from PIL import Image

import config
import json
import os.path
import requests

app = Flask(__name__)
app.config.from_object(config)

url = 'http://127.0.0.1:5000'

test_transform = tfs.Compose([
    tfs.Resize([224,224]),
    tfs.ToTensor(),
    tfs.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# 定义模型
model = models.resnet50()
model.fc = nn.Linear(2048, 3)

# 加载模型参数
params=torch.load('./model/save_model.pth')
model.load_state_dict(params)

classes_to_idx = {'非法小广告': 0, '流动商贩': 1, '垃圾暴露': 2}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload/', methods=['POST'])
def uploadiamge():
    '''
    save a pic on py project from local pc
    :return:
    '''
    file = request.files['filechoose']
    file.save('./tmp/test.jpg')
    return use_recog_api()


def use_recog_api():
    module_path = os.path.dirname(__file__)
    img = Image.open(r'./tmp/test.jpg')  # 获得用户上传的图片
    img = test_transform(img)  # 变换图片到合适的输入格式，resize，变成Tensor，减去均值
    img = torch.unsqueeze(img, dim=0)  # 增加一个维度，满足模型的输入维度(1,3,224,224)
    model.eval()  # 模型启动测试模式，该模式不使用BN层和Dropout
    out = model(img)  # 把图片输入模型后，得到分类结果
    _, pred = out.max(1)  # 最大的结果就是对应的label
    result = list (classes_to_idx.keys()) [list (classes_to_idx.values()).index (pred.item())]  # 获得label的标签描述
    return result

app.run()
    








