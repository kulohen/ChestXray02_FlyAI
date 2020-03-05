# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import psutil
import os
import argparse
from time import clock
from net import Net
import keras
from flyai.dataset import Dataset
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from model import Model
from path import MODEL_PATH
import WangyiUtilOnFlyai
from WangyiUtilOnFlyai import DatasetByWangyi,historyByWangyi,OptimizerByWangyi
from keras.engine.saving import load_model

import tensorflow as tf
from keras import backend as K


# 导入flyai打印日志函数的库
from flyai.utils.log_helper import train_log

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
下载模版之后需要把当前样例项目的app.yaml复制过去哦～
第一次使用请看项目中的：FLYAI项目详细文档.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=28, type=int, help="batch size")
args = parser.parse_args()

num_classes = 4
val_batch_size = {
    0: 33,
    1: 17,
    2: 16,
    3: 34
}
train_epoch = args.EPOCHS
history_train = 0
history_test = 0
best_score_by_acc = 0.
best_score_by_loss = 999.
lr_level=0
# 训练集的每类的batch的量，组成的list
train_batch_List = [100] * num_classes

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
# wangyi.ReadFileNames()
dataset_wangyi = DatasetByWangyi(num_classes)
dataset_wangyi.set_Batch_Size(train_batch_List, val_batch_size)
myhistory = historyByWangyi()


'''
实现自己的网络机构
'''
time_0 = clock()
# 创建最终模型

model_cnn = Net(num_classes=num_classes).get_Model()
# model_cnn = keras_model(inputs=Inp, outputs=predictions)

# 输出模型的整体信息
model_cnn.summary()

model_cnn.compile(loss='categorical_crossentropy',
              optimizer=OptimizerByWangyi().get_create_optimizer(name='adam', lr_num=1e-3),
              metrics=['accuracy']
              )

print('keras model,compile, 耗时：%.1f 秒' % (clock() - time_0))

for epoch in range(train_epoch):

    '''
    1/ 获取batch数据
    '''
    x_3, y_3, x_4, y_4, x_5, y_5 = dataset_wangyi.get_Next_Batch()
    if x_3 is None:
        cur_step = str(epoch + 1) + "/" + str(train_epoch)
        print('\n步骤' + cur_step, ': 无batch 跳过此次循环')
        continue
    # 采用数据增强ImageDataGenerator
    datagen = ImageDataGenerator(
        # rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    # datagen.fit(x_train_and_x_val)
    data_iter_train = datagen.flow(x_3, y_3, batch_size=args.BATCH, save_to_dir=None)
    # 打印步骤和训练集/测试集的量
    cur_step = str(epoch + 1) + "/" + str(train_epoch)
    print('\n步骤' + cur_step, ': %d on train, %d on val' % (len(x_3), len(x_4)))
    '''
    2/ 训练train，验证val
    '''
    # history_train = model_cnn.fit(x=x_3, y=y_3, validation_data=(x_4, y_4),
    #                                         batch_size=args.BATCH ,epochs=1,verbose=2
    #                               )
    # print('np.sum(train_batch_List :',np.sum(train_batch_List))
    for_fit_generator_train_steps = int(np.sum(train_batch_List, axis=0) * 2 / args.BATCH)
    print('该epoch的fit_generator steps是 ', for_fit_generator_train_steps)
    history_train = model_cnn.fit_generator(
        generator=data_iter_train,
        steps_per_epoch=for_fit_generator_train_steps,
        validation_data=(x_4, y_4),
        validation_steps=for_fit_generator_train_steps,
        epochs=1,
        verbose=2
    )
    history_train_all = myhistory.SetHistory(history_train)

    # 每10 epoch 重置了wangyi.dataset，防止内存泄露
    if psutil.virtual_memory().percent > 90:
        print('内存占用率：', psutil.virtual_memory().percent,'现在启动model_cnn重置')
        tmp_model_path = os.path.join(os.curdir, 'data', 'output', 'model','reset_model_tmp.h5')
        model_cnn.save(tmp_model_path)  # creates a HDF5 file 'my_model.h5'
        del model_cnn  # deletes the existing model
        model_cnn =load_model(tmp_model_path)
        print('已重置了del model_cnn，防止内存泄露')
    elif psutil.virtual_memory().percent > 80:
        print('内存占用率：', psutil.virtual_memory().percent, '，将在90%重置model_cnn')

    sum_loss = 0
    sum_acc = 0
    for iters in range(num_classes):
        if dataset_wangyi.dataset_slice[iters].get_train_length() == 0 or dataset_wangyi.dataset_slice[
            iters].get_validation_length() == 0:
            continue
        history_test = model_cnn.evaluate(
            x=x_5[iters],
            y=y_5[iters],
            batch_size=None,
            verbose=2
        )

        # 不打印了，显示的界面篇幅有限
        print('class-%d __ loss :%.4f , acc :%.4f' % (iters, history_test[0], history_test[1]))
        sum_loss += history_test[0] * val_batch_size[iters]
        sum_acc += history_test[1] * val_batch_size[iters]
        '''
         2.3修改下一个train batch
         
        # val-loss 0.7以下不提供batch, 0.7 * 20 =14
        next_train_batch_size = int(history_test[0] * 20)
        # next_train_batch_size = int(history_test[0] * val_batch_size[iters])
        # next_train_batch_size = history_test[0] + train_allow_loss[iters]
        # next_train_batch_size = int (next_train_batch_size * val_batch_size[iters])
        if next_train_batch_size > 50:
            train_batch_List[iters] = next_train_batch_size =50
        elif next_train_batch_size < 1:
            train_batch_List[iters] = next_train_batch_size= 1
        else:
            train_batch_List[iters] = next_train_batch_size
         
        '''
        train_batch_List = [
            100,100,100,100
        ]




    dataset_wangyi.set_Batch_Size(train_batch_List, val_batch_size)
    # sum_loss =sum_loss / np.sum(train_batch_List, axis = 0)
    # sum_acc = sum_acc / np.sum(train_batch_List, axis=0)

    '''
    3/ 保存最佳模型model
    '''
    # save best acc
    if history_train.history['val_acc'][0] > 0.50 and \
            round(best_score_by_loss, 2) >= round(history_train.history['val_loss'][0], 2):
    # if history_train.history['acc'][0] > 0.6 and \
    #         round(best_score_by_acc, 2) <= round(history_train.history['val_acc'][0], 2):
        model.save_model(model=model_cnn, path=MODEL_PATH, overwrite=True)
        best_score_by_acc = history_train.history['val_acc'][0]
        best_score_by_loss = history_train.history['val_loss'][0]
        print('【保存】了最佳模型by val_loss : %.4f' % best_score_by_loss)


    # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
    # train_log(train_loss=history_train.history['loss'][0], train_acc=history_train.history['acc'][0], val_loss=history_train.history['val_loss'][0], val_acc=history_train.history['val_acc'][0])
    # train_log(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc)
    # train_log(train_loss=loss.item(), train_acc=train_accuracy)

    '''
    4/ 调整学习率和优化模型
    '''
    tmp_opt = None
    if epoch == 0 or epoch==50 or epoch==100 or epoch==150 :
        pass
    elif epoch % 50 ==0:
        tmp_opt = OptimizerByWangyi().get_random_opt()

    # 调整学习率，且只执行一次
    if history_train.history['loss'][0] < 0.8 and lr_level == 0:

        tmp_opt = OptimizerByWangyi().get_create_optimizer(name='adagrad', lr_num=1e-4)
        lr_level = 1

    elif history_train.history['loss'][0] < 0.6 and lr_level == 1:
        tmp_opt = OptimizerByWangyi().get_create_optimizer(name='sgd', lr_num=1e-5)
        lr_level = 2

    elif history_train.history['loss'][0] < 0.3 and lr_level == 2:
        tmp_opt = tmp_opt = OptimizerByWangyi().get_create_optimizer(name='sgd', lr_num=1e-4)
        lr_level = 3

    elif history_train.history['loss'][0] < 0.1 and lr_level == 3:
        tmp_opt = tmp_opt = OptimizerByWangyi().get_create_optimizer(name='adagrad', lr_num=1e-5)
        lr_level = 4

    # 应用新的学习率
    if tmp_opt is not None:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=tmp_opt,
                          metrics=['accuracy'])

    # TODO 新的学习率，还没完成
    # if optimzer_custom.compareHistoryList( history_train_all['loss'] ,pationce= 5 ,min_delta=0.001) :
    #     model_cnn.compile(loss='categorical_crossentropy',
    #                       optimizer=optimzer_custom.get_next() ,
    #                       metrics=['accuracy'])
    #TODO 动态冻结训练层？

    '''
    5/ 冻结训练层
    '''

print('best_score_by_acc :%.4f' % best_score_by_acc)
print('best_score_by_loss :%.4f' % best_score_by_loss)