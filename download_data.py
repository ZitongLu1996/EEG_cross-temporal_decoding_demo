# -*- coding: utf-8

"""
@File       :   download_data.py
@Author     :   Zitong Lu
@Contact    :   zitonglu1996@gmail.com
@License    :   MIT License
"""

import os
from six.moves import urllib
from pyctrsa.util.download_data import schedule
from pyctrsa.util.unzip_data import unzipfile

# 下载用于decoding的示例数据
url = 'https://attachment.zhaokuangshi.cn/BaeLuck_2018jn_data_ERP_5subs.zip'
filename = 'BaeLuck_2018jn_data_ERP_5subs.zip'
data_dir = 'data/'
classification_results_dir = 'classification_results/'
filepath = data_dir + filename
exist = os.path.exists(filepath)
if exist == False:
    os.makedirs(data_dir)
    urllib.request.urlretrieve(url, filepath, schedule)
    print('Download completes!')
elif exist == True:
    print('Data already exists!')

# 压缩下载的示例数据
unzipfile(filepath, data_dir)