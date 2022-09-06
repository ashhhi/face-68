import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,GlobalMaxPool2D,Dense,BatchNormalization,Activation,GlobalAvgPool2D,Flatten
from tensorflow.keras import Model
import cv2 as cv
from Utils import DataLoader,Save_history,Visual

#True为训练，False为测试


isTrain = False

tf.config.experimental_run_functions_eagerly(True)
SEED = 3253415343
STDDEV = 0.1
IMAGE_DATA_TYPE = tf.float32
NUM_CHANNELS = 3 #图像的通道数
BATCH_SIZE = 16
EPOCH = 50
N_POINTS = 68
IMAGE_SIZE = 224
LEARNING_RATE = 0.001


x_train,y_train,x_test,y_test = DataLoader()

print("---------------------数据预处理完成---------------------")
# print(x_train)
# print(y_train)
y_train = tf.cast(y_train,tf.float32)
y_test = tf.cast(y_test,tf.float32)
#通道注意力模块
class Channal(Model):
    def __init__(self,out_dim,ratio=16):
        super(Channal, self).__init__()
        self.out_dim = out_dim
        self.ratio = ratio
        self.d1 = Dense(units=out_dim / self.ratio)
        self.a1 = Activation('relu')
        self.d2 = Dense(units=out_dim)
        self.a2 = Activation('sigmoid')

    def call(self,input_x):
        squeeze1 = GlobalAvgPool2D()(input_x)

        excitation1 = self.d1(squeeze1)
        excitation1 = self.a1(excitation1)
        excitation1 = self.d2(excitation1)
        excitation1 = self.a2(excitation1)
        excitation1 = tf.reshape(excitation1, [-1, 1, 1,self.out_dim])

        squeeze2 = GlobalMaxPool2D()(input_x)
        excitation2 = self.d1(squeeze2)
        excitation2 = self.a1(excitation2)
        excitation2 = self.d2(excitation2)
        excitation2 = self.a2(excitation2)
        excitation2 = tf.reshape(excitation2, [-1, 1, 1, self.out_dim])
        excitation = excitation1 + excitation2
        scale = input_x * excitation
        return scale

#空间注意力模块
class Spatial(Model):
    def __init__(self):
        super(Spatial, self).__init__()
        self.c1 = Conv2D(kernel_size=3,filters=1,strides=1,padding='same')
        self.a1 = Activation('sigmoid')
    def call(self,input_x):
        # print("input_xshape",input_x.shape)
        x1 = tf.reduce_max(input_x,3)
        x2 = tf.reduce_mean(input_x,3)
        # print("input_xshape1", x1.shape)
        # print("input_xshape2", x2.shape)
        x1 = tf.reshape(x1,(x1.shape[0],x1.shape[1],x1.shape[2],1))
        x2 = tf.reshape(x2,(x2.shape[0],x2.shape[1],x2.shape[2],1))
        # print("input_xshape1", x1.shape)
        # print("input_xshape2", x2.shape)
        x = tf.concat((x1,x2),axis=3)
        # print(x.shape)

        x = self.c1(x)
        x = self.a1(x)
        y = x * input_x
        # print(y.shape)
        return y

class CBAM(Model):
    def __init__(self,initial_filters):
        super(CBAM, self).__init__()
        self.initial_filters = initial_filters
        #通道注意力模块
        self.ca = Channal(initial_filters)
        self.sa = Spatial()
    def call(self,input_x):
        x = self.ca(input_x)
        x = self.sa(x)
        return x

class ResNetBlock(Model):
    def __init__(self,filters,strides=1,residual_path=False):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters,(3,3),strides=strides,padding='same',use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')


        self.c2 = Conv2D(filters,(3,3),strides=1,padding='same',use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path == True:
            self.down_c1 = Conv2D(filters,(1,1),strides=strides,padding='same',use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')


    def call(self,inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(residual)
            residual = self.down_b1(residual)

        out = self.a2(y+residual)
        return out

class Resnet18(Model):
    def __init__(self,initial_filters=64,out_filters = N_POINTS * 2):
        super(Resnet18, self).__init__()
        self.c1 = Conv2D(initial_filters,(7,7),strides=1,padding='same',use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.drop = Dropout(0.3)

        self.attention1 = CBAM(initial_filters)

        self.block1 = ResNetBlock(initial_filters)
        self.block2 = ResNetBlock(initial_filters)

        self.block3 = ResNetBlock(initial_filters*2,residual_path=True,strides=2)
        self.block4 = ResNetBlock(initial_filters*2)

        self.block5 = ResNetBlock(initial_filters*4, residual_path=True,strides=2)
        self.block6 = ResNetBlock(initial_filters*4)

        self.block7 = ResNetBlock(initial_filters*8, residual_path=True,strides=2)
        self.block8 = ResNetBlock(initial_filters*8)

        self.attention2 = CBAM(initial_filters*8)
        self.f = Flatten()
        self.d1 = Dense(out_filters*2)
        self.d2 = Dense(out_filters)

    def call(self,inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.attention1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.attention2(x)

        x = self.f(x)
        x = self.d1(x)
        y = self.d2(x)
        y = tf.reshape(y,(y.shape[0],y.shape[1]//2,2))
        return y


model = Resnet18()

def LOSS(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss



#指数衰减学习率
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.0001, decay_steps=1000, decay_rate=0.96)
opt = tf.keras.optimizers.Adam(exponential_decay)



checkpoint_save_path = 'CKPT/Res_CBAM_Wingloss_CKPT\Resnet.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
												save_weights_only = True,
												save_best_only = True
												)
if os.path.exists(checkpoint_save_path+'.index'):
	print("---------------------加载模型成功---------------------")
	model.load_weights(checkpoint_save_path)


if isTrain == True:
    print("---------------------开始训练模型---------------------")


    model.compile(optimizer=opt,
                  loss = LOSS,
                  metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH,validation_data=(x_test,y_test),callbacks=[cp_callback])
    model.summary()
    Save_history('log/Res_CBAM.csv', model.history)
    Visual(model.history)


else:
    for i in range(500, 510):
        img = x_train[i]
        test = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
        pts = model.predict(test)
        pts = pts.reshape((N_POINTS, 2))
        print(pts)
        for i in range(len(pts)):
            cv.circle(img, (int(pts[i][0]), int(pts[i][1])), 1, (0, 255, 0))
        cv.imshow("pred_landmark", img)
        cv.waitKey()
