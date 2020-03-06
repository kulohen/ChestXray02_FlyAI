## build CNN
from keras.applications import ResNet50,VGG16,InceptionResNetV2,DenseNet121,DenseNet201
from keras.layers import Input,Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
from keras.models import Model as keras_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint,ReduceLROnPlateau
from keras.optimizers import SGD,Adam,RMSprop
from processor import img_size
from flyai.utils import remote_helper
import psutil
import os
from keras.engine.saving import load_model

class Net():

    def __init__(self, num_classes=10):
        """Declare all needed layers."""
        self.num_classes = num_classes
        try:
            weights_path =None
            # weights_path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
            weights_path = remote_helper.get_remote_data(
                'https://www.flyai.com/m/v0.8|densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
            # weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
        except OSError:
            weights_path = 'imagenet'


        # base_model = ResNet50(weights=None, input_shape=input_shape=(img_size[0], img_size[1], 3), include_top=False)
        base_model = DenseNet121(weights=weights_path, include_top=False ,input_shape=(img_size[0], img_size[1],3))

        Inp = Input(shape=(img_size[0], img_size[1],3))

        # x = Conv2D(256,3,
        #                   activation='relu',
        #                   padding='same',
        #                   name='wangyi_conv1')(Inp)
        # x = Conv2D(256,5,
        #                   activation='relu',
        #                   padding='same',
        #                   name='wangyi_conv2')(x)
        # x = MaxPooling2D((2, 2), strides=(1, 1), name='wangyi_pool')(x)
        # x =Flatten()(x)
        # x = Conv2D(3,7,
        #                   activation='relu',
        #                   padding='same',
        #                   name='wangyi_conv3')(x)
        # 增加定制层
        x = base_model(Inp)
        # x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        # x = Flatten(name='flatten_1')(x)

        # 冻结不打算训练的层。
        # print('base_model.layers', len(base_model.layers))
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)
        #
        # for layer in base_model.layers[:]:
        #     layer.trainable = False
        # print(layer)

        x = GlobalAveragePooling2D()(x)
        # x = Flatten(name='flatten_1')(x)
        # x = Dense(2048, activation='relu' )(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        # 创建最终模型

        self.model_cnn = keras_model(inputs=Inp, outputs=predictions)


    def get_Model(self):
        return self.model_cnn

    def cleanMemory(self):
        if psutil.virtual_memory().percent > 90:
            print('内存占用率：', psutil.virtual_memory().percent, '现在启动model_cnn重置')
            tmp_model_path = os.path.join(os.curdir, 'data', 'output', 'model', 'reset_model_tmp.h5')
            self.model_cnn.save(tmp_model_path)  # creates a HDF5 file 'my_model.h5'
            del self.model_cnn  # deletes the existing model
            self.model_cnn = load_model(tmp_model_path)
            print('已重置了del model_cnn，防止内存泄露')
        elif psutil.virtual_memory().percent > 80:
            print('内存占用率：', psutil.virtual_memory().percent, '%，将在90%重置model_cnn')

if __name__=='__main__':
    Net().get_Model().summary()
    # x = VGG16()
    # x.summary()