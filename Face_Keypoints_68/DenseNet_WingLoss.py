import math
import os
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import Model
from Utils import DataLoader,Visual,Save_history
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,AvgPool2D,Dense,Flatten

# tf.config.experimental_run_functions_eagerly(True)
x_train,y_train,x_test,y_test = DataLoader()
y_train = tf.cast(y_train,tf.float32)
y_test = tf.cast(y_test,tf.float32)
assert y_train[0].shape == y_test[0].shape,"训练集和测试集的数据形状不同"

#定义宏
isTrain = True
IMAGE_SIZE = x_train.shape[1]
BATCH_SIZE = 2
growth_rate = 12
N_POINTS = 68
EPOCH = 50

class DenseLayer(Model):
    def __init__(self,growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c1 = Conv2D(bn_size * growth_rate,kernel_size=1,strides=1,use_bias=False,padding='same')

        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.c2 = Conv2D(growth_rate, kernel_size=3, strides=1, use_bias=False,padding='same')
    def call(self,input_x):
        x = self.b1(input_x)
        x = self.a1(x)
        x = self.c1(x)

        x = self.b2(x)
        x = self.a2(x)
        y = self.c2(x)
        return y

class DenseBlock(Model):
    def __init__(self,num_layers,bn_size,growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.Layers = []
        for i in range(num_layers):
            layer = DenseLayer(growth_rate,bn_size)
            self.Layers.append(layer)
    def call(self,x):
        pre_input = []
        for i in range(self.num_layers):
            if i == 0:
                pre_input = x
            else:
                pre_input = tf.concat([pre_input,x],axis=3)
            x = self.Layers[i](x)
            x = tf.concat([pre_input,x],axis=3)
        return x

class Transition(Model):
    def __init__(self,input_feature,drop_rate=1.0):
        super(Transition, self).__init__()
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c1 = Conv2D(input_feature*drop_rate,kernel_size=1,strides=1,use_bias=False)
        self.p1 = AvgPool2D(2,strides=2)
    def call(self,x):
        x = self.b1(x)
        x = self.a1(x)
        x = self.c1(x)
        y = self.p1(x)
        return y

class DenseNet(Model):
    def __init__(self,input_feature,num_DenseBlock,num_layers):
        super(DenseNet, self).__init__()
        self.num_DenseBlock = num_DenseBlock

        self.c1 = Conv2D(3,kernel_size=7,strides=1,padding='same',use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = AvgPool2D(2,strides=2)

        self.Module = tf.keras.models.Sequential()

        for i in range(num_DenseBlock):
            self.Module.add(DenseBlock(num_layers=num_layers,bn_size=BATCH_SIZE,growth_rate=growth_rate))
            input_feature += growth_rate * num_layers
            if i != num_DenseBlock-1:
                self.Module.add(Transition(input_feature,drop_rate=0.5))

        self.f = Flatten()
        self.d = Dense(N_POINTS*2)

    def call(self,x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        # print("Module前",x.shape)
        x = self.Module(x)
        # print("Module后",x.shape)
        x = self.f(x)
        y = self.d(x)
        y = tf.reshape(y,(y.shape[0],y.shape[1]//2,2))
        return y

#input_feature就是图像初始的通道数，num_DenseBlock就是block的个数，
model = DenseNet(input_feature=3,num_DenseBlock=3,num_layers=5)

# def LOSS(y_true, y_pred):
#     y_true = tf.cast(y_true, dtype=tf.float32)
#     return tf.nn.l2_loss(tf.subtract(y_pred ,y_true))/(2*N_POINTS*BATCH_SIZE)
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

checkpoint_save_path = 'CKPT/DenseNet_Wingloss_CKPT\DenseNet.ckpt'
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
    Save_history('log_temp/DenseNet_Wingloss_log.csv', model.history)
    Visual(model.history)


else:
    for i in range(250, 260):
        img = x_train[i]
        test = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
        pts = model.predict(test)
        pts = pts.reshape((N_POINTS, 2))
        print(pts)
        for i in range(len(pts)):
            cv.circle(img, (int(pts[i][0]), int(pts[i][1])), 1, (0, 255, 0))
        cv.imshow("pred_landmark", img)
        cv.waitKey()