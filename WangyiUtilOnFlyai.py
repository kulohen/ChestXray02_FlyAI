#!/usr/bin/env python
# coding:utf-8
"""
Name : WangyiUtilOnFlyai.py
Author  : 莫须有的嚣张
Contect : 291255700
Time    : 2019/7/28 上午9:42
Desc:
"""
import psutil
import os
from time import clock
import time

import keras.optimizers as optmzs
import numpy as np

import pandas as pd
from flyai.core import Lib
from flyai.dataset import Dataset

from flyai.source.base import DATA_PATH
from flyai.source.csv_source import Csv
import csv
from model import Model


lr_level = {
            0:0.001,
            1:0.0003,
            2:0.0001,
            3:3e-5,
            4:1e-5
        }
optimizer_level = {
    0: 'sgd',
    1: 'rmsprop',
    2: 'adagrad',
    3: 'adadelta',
    4: 'adam',
    5: 'adamax',
    6: 'nadam'
}
optimizer_name = {
            'sgd' : optmzs.SGD,
            'rmsprop': optmzs.RMSprop,
            'adagrad' : optmzs.Adagrad,
            'adadelta' : optmzs.Adadelta,
            'adam' : optmzs.Adam,
            'adamax' : optmzs.Adamax,
            'nadam' : optmzs.Nadam
        }
class OptimizerByWangyi():
    def __init__(self, pationce=5 , min_delta =0.003):
        self.optimizer_iterator = 0
        self.lr_iterator = 0
        self.pationce_count = 0

    def get_create_optimizer(self,name=None,lr_num=0):
        if name is None or lr_num<=0:
            raise ValueError('请指定正确的优化器/学习率')

        x = optimizer_name[name](lr=lr_num)
        print('采用了优化器： ',name , ' --学习率: ', lr_num)
        return x

    def get_next(self, optimzer = None, lr = None):

        if optimzer is not None:
            name_1 = optimzer
        else:
            name_1 = optimizer_level[self.optimizer_iterator]

        if lr is not None:
            lr_1 = lr
        else:
            lr_1 = lr_level[self.lr_iterator]

        x = self.get_create_optimizer(name_1 , lr_1 )

        self.lr_iterator = (self.lr_iterator + 1) % len(lr_level)
        if self.lr_iterator == 0 :
            self.optimizer_iterator = (self.optimizer_iterator +1 ) % len(optimizer_level)
        return x
    #TODO 写降低学习率的回调功能？
    def compareHistoryList(self, loss_list =None , pationce = 10, min_delta =0.001):
        '''
        判断这个自定义的callback，达到earlystopping , reduceLearnRate 的条件 返回True
        :param loss_list:
        :param pationce:
        :param min_delta:
        :return:
        '''
        self.pationce_count += 1
        if loss_list is None :
            ValueError('第一个参数不能为空')
        elif len(loss_list) < pationce or self.pationce_count <pationce:
            return  False
        #TODO 写上判断的逻辑，或者叫做callback

        # elif  (loss_list[-pationce]*pationce -sum(loss_list[-pationce: ]) ) <\
        elif (sum(loss_list[-pationce:-pationce/2]) - sum(loss_list[-pationce/2:])) < \
                  (min_delta * pationce) :
            print('um(loss_list[-pationce:-pationce/2]) - sum(loss_list[-pationce/2:])) <(min_delta * pationce ：',
                  sum(loss_list[-pationce:-pationce/2]) ,' - ',sum(loss_list[-pationce/2:]),' < ' , min_delta * pationce)
            self.pationce_count = 0
            return True
        else:
            return False

    def get_random_opt(self):
        name = np.random.randint(0, 6)
        lr = np.random.randint(0, 5)
        print('启动随机de学习率')
        return self.get_create_optimizer(optimizer_level[name] , lr_level[lr])




def readCustomCsv_V3(train_csv_url, test_csv_url):
    # 2019-08-29 flyai改版本了，这是为了适应
    # source_csv = Csv({'train_url': os.path.join(DATA_PATH, train_csv_url),
    #                                        'test_url': os.path.join(DATA_PATH, test_csv_url)})

    # 2020-3-3 flyai改版本了，这是为了适应
    source_csv = Csv({'train_url': train_csv_url,
                                           'test_url': test_csv_url})

    return source_csv


def get_sliceCSVbyClassify_V2(label='label',classify_count=3):
    # 2019-08-29 flyai改版本了，这是为了适应
    try:
        source_csv=readCustomCsv_V3('train.csv', 'test.csv')
        print('train.csv , test.csv 读取成功')
    except:
        print('train.csv , test.csv 读取失败')
        source_csv = None

    if source_csv is None:
        try:
            source_csv = readCustomCsv_V3('dev.csv', 'dev.csv')
        except:
            print('train.csv,test.csv,dev.csv 都读取失败')


    # step 1 : csv to dataframe
    dataframe_train = pd.DataFrame(data=source_csv.c.data)
    dataframe_test = pd.DataFrame(data=source_csv.c.val)


    # step 2 : 筛选 csv


    list_path_train,list_path_test = [],[]
    for epoch in range(classify_count):
        path_train = os.path.join(DATA_PATH, 'wangyi-train-classfy-' + str(epoch) + '.csv')
        a = dataframe_train[dataframe_train[label] == epoch]
        a=a.sample(frac=1)
        a.to_csv(path_train,index=False)
        list_path_train.append(path_train)

        path_test = os.path.join(DATA_PATH, 'wangyi-test-classfy-' + str(epoch) + '.csv')
        b = dataframe_test[dataframe_test[label] == epoch]
        b = b.sample(frac=1)
        b.to_csv(path_test,index=False)
        list_path_test.append(path_test)

        print('classfy-',epoch,' : train and test.csv save OK!')

    return list_path_train,list_path_test

def get_sliceCSVbyClassify_V3(label='labels',classify_count=3, split=0.8):
    # 2019-08-29 flyai改版本了，这是为了适应
    # 将train val data 合并，再重新划分。分类到对应csv
    try:
        source_csv=readCustomCsv_V3('train.csv', 'test.csv')
        print('train.csv , test.csv 读取成功')
    except:
        print('train.csv , test.csv 读取失败')
        source_csv = None

    if source_csv is None:
        try:
            source_csv = readCustomCsv_V3('dev.csv', 'dev.csv')
            print('dev.csv 读取成功')
        except:
            print('train.csv,test.csv,dev.csv 都读取失败')


    # step 1 : csv to dataframe
    dataframe_train = pd.DataFrame(data=source_csv.c.data)
    dataframe_test = pd.DataFrame(data=source_csv.c.val)

    # train and test merge one, and split by myself
    tmp_a = pd.concat([dataframe_train,dataframe_test],axis=0)
    tmp_b = tmp_a.sample(frac=1)

    # step 2 : 筛选 csv


    list_path_train,list_path_test = [],[]
    for epoch in range(classify_count):
        tmp_c = tmp_b[tmp_b[label]==epoch]
        cut_length = int(len(tmp_c) * split)
        a = tmp_c[ : cut_length]
        b = tmp_c[ cut_length : ]

        # path_train = os.path.join(DATA_PATH, 'wangyi-train-classfy-' + str(epoch) + '.csv')
        path_train = 'wangyi-train-classfy-' + str(epoch) + '.csv'
        a.to_csv(os.path.join(DATA_PATH, path_train),index=False)
        list_path_train.append(path_train)

        # path_test = os.path.join(DATA_PATH, 'wangyi-test-classfy-' + str(epoch) + '.csv')
        path_test = 'wangyi-test-classfy-' + str(epoch) + '.csv'
        b.to_csv(os.path.join(DATA_PATH,path_test) ,index=False)
        list_path_test.append(path_test)

        print('classfy-',epoch,' : train and test.csv save OK!')

    return list_path_train,list_path_test


def getDatasetListByClassfy_V4(classify_count=3):
    # 2019-08-29 flyai改版本了，这是为了适应

    xx, yy = get_sliceCSVbyClassify_V3(classify_count=classify_count,split= 0.9)
    list_tmp=[]
    for epoch in range(classify_count):
        time_0 = clock()
        dataset = Lib(source=readCustomCsv_V3(xx[epoch], yy[epoch]), epochs=1)
        list_tmp.append(dataset)
        # print('class-',epoch,' 的flyai dataset 建立成功')
        print('class-', epoch, ' 的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0), '; train_length:',
              dataset.get_train_length(), '; val_length:', dataset.get_validation_length())

    return list_tmp

class historyByWangyi():
    '''
    总结main.py中使用的代码，规整成1个类，方便调用
    '''
    def __init__(self):
        # 自定义history
        self.history_train_all = {}
        self.history_train_loss = []
        self.history_train_acc = []
        self.history_train_val_loss = []
        self.history_train_val_acc = []

    def SetHistory(self,history_train):

        self.history_train_loss.append(history_train.history['loss'][0])
        # self.history_train_acc.append(history_train.history['accuracy'][0])
        self.history_train_acc.append(history_train.history['acc'][0])
        self.history_train_val_loss.append(history_train.history['val_loss'][0])
        # self.history_train_val_acc.append(history_train.history['val_accuracy'][0])
        self.history_train_val_acc.append(history_train.history['val_acc'][0])
        self.history_train_all['loss'] = self.history_train_loss
        self.history_train_all['accuracy'] = self.history_train_acc
        self.history_train_all['val_loss'] = self.history_train_val_loss
        self.history_train_all['val_accuracy'] = self.history_train_val_acc

        return self.history_train_all

class DatasetByWangyi():
    def __init__(self, n):
        self.num_classes= n
        self.dropout = 0.5

        time_0 = clock()
        self.dataset_slice = getDatasetListByClassfy_V4(classify_count=n)
        self.optimzer_custom = OptimizerByWangyi()
        print('全部分类的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0))

        # 平衡输出45类数据
        # self.x_3, self.y_3, self.x_4, self.y_4 = [], [], [], []
        # self.x_5, self.y_5 ,self.x_6,self.y_6= {}, {} , {}, {}
        self.train_batch_List = []
        self.val_batch_size = {}

    def set_Batch_Size(self,train_size,val_size):
        self.train_batch_List = train_size
        self.val_batch_size = val_size

    def get_Next_Batch(self):
        # 平衡输出45类数据

        x_3, y_3 =self.get_Next_Train_Batch()
        x_4, y_4, x_5, y_5 =self.get_Next_Val_Batch()
        return x_3, y_3, x_4, y_4 ,x_5, y_5

    def get_Next_Train_Batch(self):
        # 平衡输出45类数据
        x_3, y_3 = [], []

        for iters in range(self.num_classes):
            if self.dataset_slice[iters].get_train_length() == 0 or self.train_batch_List[iters] == 0:
                continue
            tmp_size = int(self.train_batch_List[iters] / self.dropout )

            xx_tmp_train, yy_tmp_train,_,_= self.dataset_slice[iters].next_batch(size=tmp_size, test_size=0)
            #TODO dropout 0.5的量
            tmp_size = len(xx_tmp_train)
            per_2 = np.random.permutation(tmp_size)  # 打乱后的行号
            xx_tmp_train = xx_tmp_train[per_2, :, :, :]  # 获取打乱后的训练数据
            yy_tmp_train = yy_tmp_train[per_2, :]
            xx_tmp_train = xx_tmp_train[0:tmp_size]
            yy_tmp_train = yy_tmp_train[0:tmp_size]

            x_3.append(xx_tmp_train)
            y_3.append(yy_tmp_train)


        # 跳出当前epoch，貌似不需要这个if
        if len(x_3) == 0 or len(y_3) == 0 :
            return None
        x_3 = np.concatenate(x_3, axis=0)
        y_3 = np.concatenate(y_3, axis=0)

        # shuffle train-data
        per = np.random.permutation(len(x_3))  # 打乱后的行号
        x_3 = x_3[per, :, :, :]  # 获取打乱后的训练数据
        y_3 = y_3[per, :]

        return x_3, y_3

    def get_Next_Val_Batch(self):
        # 平衡输出45类数据
        x_4, y_4 = [], []
        x_5, y_5 = {}, {}

        for iters in range(self.num_classes):
            if self.dataset_slice[iters].get_validation_length() == 0:
                continue
            # xx_tmp_val, yy_tmp_val = self.dataset_slice[iters].next_validation_batch(size = self.val_batch_size[iters])
            _, _, xx_tmp_val, yy_tmp_val = self.dataset_slice[iters].next_batch(size=0, test_size=self.val_batch_size[iters])
            # 合并3类train

            x_4.append(xx_tmp_val)
            y_4.append(yy_tmp_val)
            x_5[iters] = xx_tmp_val
            y_5[iters] = yy_tmp_val

        # 跳出当前epoch，貌似不需要这个if
        if len(x_4) == 0 or len(y_4) == 0:
            return None,None,None,None

        x_4 = np.concatenate(x_4, axis=0)
        y_4 = np.concatenate(y_4, axis=0)
        return x_4, y_4 ,x_5, y_5

    def read_predict_Csv(self,filename='model.h5'):
        PREDICT_DATA_PATH = os.path.join(os.curdir, 'data', 'input','upload.csv')
        Csvlist = []
        if os.path.exists(PREDICT_DATA_PATH):
            print(PREDICT_DATA_PATH)
        else:
            print('do not exit')
        with open(PREDICT_DATA_PATH, 'r') as csvFile:
            reader = csv.reader(csvFile)
            headers = next(reader)  # 获得列表对象，包含标题行的信息
            print(headers)

            for row in reader:  # 循环打印各行内容
                Csvlist.append(row[0])
        print('Csvlist.count is ', len(Csvlist))
        return Csvlist

    #TODO 没实现
    def write_predict_Csv(self, filename='文件名.csv'):
        # 1. 创建文件对象
        f = open(filename, 'w', encoding='utf-8')

        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)

        # 3. 构建列表头
        csv_writer.writerow(["image_path", "labels"])

        # 4. 写入csv文件内容
        csv_writer.writerow(["l", '18', '男'])
        csv_writer.writerow(["c", '20', '男'])
        csv_writer.writerow(["w", '22', '女'])


    def predict_to_csv(self):
        save_file_name = 'upload-by-' + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.csv')
        save_file_name = os.path.join(os.curdir, 'data', 'output', save_file_name)

        # 1. 创建文件对象
        with open(save_file_name, 'w', encoding='utf-8',newline='' "") as f:


            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)

            # 3. 构建列表头
            csv_writer.writerow(["image_path", "labels"])

            url_list = self.read_predict_Csv()
            dataset = Dataset(epochs=5, batch=16)
            model = Model(dataset)
            predict_list = []
            for row in url_list:
                predict_num = model.predict(image_path=row)
                # csv_writer.writerow([row, str(predict_num)])
                # predict_list.append(predict_num)
                predict_list.append([row,predict_num])
                # 打印进度条
                count_now = len(predict_list)
                count_total = len(url_list)
                print('\r'+'预测集进度：'+str(count_now)+'/'+ str(count_total),
                      '----{:.2%}'.format(count_now/count_total),end='', flush=True)
            csv_writer.writerows(predict_list)
        print('\n已保存CSV到 ',save_file_name)


def ReadFileNames():
    for root, dirs, files in os.walk(os.getcwd()+'/data/input'):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件

if __name__=='__main__':
    if psutil.virtual_memory().percent > 1:
        print(psutil.virtual_memory().percent)

