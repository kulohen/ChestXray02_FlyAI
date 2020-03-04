# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
from keras.engine.saving import load_model
from path import MODEL_PATH
from flyai.dataset import Dataset

KERAS_MODEL_NAME = "model.h5"


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)

    '''
    评估一条数据
    '''

    def predict(self, **data):
        if self.model is None:
            self.model = load_model(self.model_path)
        data = self.model.predict(self.dataset.predict_data(**data))
        data = self.dataset.to_categorys(data)
        return data

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        if self.model is None:
            self.model = load_model(self.model_path)
        labels = []
        for data in datas:
            data = self.model.predict(self.dataset.predict_data(**data))
            data = self.dataset.to_categorys(data)
            labels.append(data)

        return labels

    '''
    保存模型的方法
    '''

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))


if __name__ == '__main__':

    print('ojbk')
    dataset = Dataset(epochs=5, batch=16)
    model = Model(dataset)

    p = model.predict_all([])
    print(p)

    x,y = dataset.next_train_batch()
    a = {
        'images/00007635_001.png': 0,
        'images/00002573_000.png': 0,
        'images/00000368_005.png': 0,
    }
    a1 = {
        'images/00007635_001.png':0
    }
    aa = [
        'images/00007635_001.png',
        'images/00002573_000.png',
        'images/00000368_005.png'
    ]
    aa1 = [
        {
        'images/00007635_001.png': 0
        }
    ]
    c = model.predict( image_path='images/00001718_012.png')
    print('predict is ',c)

    # b = model.predict_all(**a1)
    #b = model.predict_all( datas =aa)

    # print(b)