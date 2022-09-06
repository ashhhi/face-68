import math
import os
import numpy as np
import tensorflow as tf
from keras import backend
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,BatchNormalization,Activation,GlobalAvgPool2D
from tensorflow.keras import Model
import cv2 as cv
from Utils import DataLoader, Save_history, Visual

isTrain = True
tf.config.experimental_run_functions_eagerly(True)
SEED = 3253415343
STDDEV = 0.1
IMAGE_DATA_TYPE = tf.float32
NUM_CHANNELS = 3 #图像的通道数
BATCH_SIZE = 1
EPOCH = 1
N_POINTS = 68
IMAGE_SIZE = 224
LEARNING_RATE = 0.001


x_train,y_train,x_test,y_test = DataLoader()
assert y_train[0].shape == y_test[0].shape,"训练集和测试集的数据形状不同"
y_train = tf.cast(y_train,tf.float32)
y_test = tf.cast(y_test,tf.float32)
print("---------------------数据预处理完成---------------------")

#SE通道注意力模块
class SElayer(Model):
    def __init__(self,out_dim,ratio=16):
        super(SElayer, self).__init__()
        self.out_dim = out_dim
        self.ratio = ratio
        self.d1 = Dense(units=out_dim / self.ratio)
        self.a1 = Activation('relu')
        self.d2 = Dense(units=out_dim)
        self.a2 = Activation('sigmoid')

    def call(self,input_x):
        squeeze = GlobalAvgPool2D()(input_x)
        excitation = self.d1(squeeze)
        excitation = self.a1(excitation)
        excitation = self.d2(excitation)
        excitation = self.a2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1,self.out_dim])
        scale = input_x * excitation
        return scale

class ResNetBlock(Model):
    def __init__(self,filters,strides=1,residual_path=False,):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters,(3,3),strides=strides,padding='same',use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.drop = Dropout(0.3)


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


        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)
        # y = self.drop(x)
        if self.residual_path:
            residual = self.down_c1(residual)
            residual = self.down_b1(residual)

        out = self.a2(y+residual)
        return out

class Resnet18(Model):
    def __init__(self,initial_filters=64,out_filters = N_POINTS * 2):
        super(Resnet18, self).__init__()
        self.c1 = Conv2D(initial_filters,(3,3),strides=1,padding='same',use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.drop = Dropout(0.3)
        self.ca1 = SElayer(initial_filters)
        self.f = Flatten()

        self.block1 = ResNetBlock(initial_filters)
        self.block2 = ResNetBlock(initial_filters)

        self.block3 = ResNetBlock(initial_filters*2,residual_path=True,strides=2)
        self.block4 = ResNetBlock(initial_filters*2)

        self.block5 = ResNetBlock(initial_filters*4, residual_path=True,strides=2)
        self.block6 = ResNetBlock(initial_filters*4)

        self.block7 = ResNetBlock(initial_filters*8, residual_path=True,strides=2)
        self.block8 = ResNetBlock(initial_filters*8)

        self.ca2 = SElayer(initial_filters*8)
        self.d1 = Dense(out_filters*2)
        self.d = Dense(out_filters)

    def call(self,inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.ca1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.ca2(x)
        x = self.f(x)
        x = self.d1(x)
        y = self.d(x)
        y = tf.reshape(y, (y.shape[0], y.shape[1] // 2, 2))
        return y


model = Resnet18()

# def LOSS(y_true, y_pred):
# 	y_true = tf.cast(y_true, dtype=tf.float32)
# 	return tf.nn.l2_loss(tf.subtract(y_pred ,y_true))/(2*N_POINTS*BATCH_SIZE)
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

checkpoint_save_path = 'CKPT/Res_SE_CKPT\Resnet.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
												save_weights_only = True,
												save_best_only = True
												)
if os.path.exists(checkpoint_save_path+'.index'):
	print("---------------------加载模型成功---------------------")
	model.load_weights(checkpoint_save_path)


print("---------------------开始训练模型---------------------")



# model.run_eagerly = True

if isTrain == True:
    model.compile(optimizer=opt,
                  loss=LOSS,
                  metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCH,validation_data=(x_test,y_test),callbacks=[cp_callback])
    model.summary()
    # Save_history('log_temp/Res_SE_Wingloss.csv', model.history)
    # Visual(model.history)

else:
    #验证
    sum_pingfang = 0
    comgregate = 0
    cnt = 0
    for index in range(0,x_train.shape[0]):
        temp = 0
        p = model.predict(x_train[index].reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))).reshape((N_POINTS,2))
        s = y_train[index].reshape((N_POINTS,2))
        for index in range(0,len(p)):
            ss = np.sqrt((p[index][0]-s[index][0])**2+(p[index][1]-s[index][1])**2)
            temp += ss

        d = np.sqrt((s[0][0]-s[1][0])**2+(s[0][1]-s[1][1])**2)
        temp = temp / d
        comgregate += temp
        cnt += 1
        print(cnt)
    comgregate /= x_train.shape[0]
    print(comgregate)
