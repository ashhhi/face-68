import numpy as np
import os
import cv2 as cv

SIZE = 224


def readLmk(fileName):
    landmarks = []
    if not os.path.exists(fileName):
        return landmarks
    else:
        fp = open(fileName)
        i = 0
        for line in fp.readlines():
            if i>2 and i<71:
                temp = line.split(" ")
                x = int(float(temp[0]))
                y = int(float(temp[1]))
                landmarks.append((x,y))
            i += 1
    return landmarks


def generator(file_path):
    if os.path.exists(file_path)==False:
        print("the file is not exist")
    else:
        pic = []
        pts = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if(file.split('.')[1]=='jpg' or file.split('.')[1]=='png'):
                    if len(file.split('.')[0].split('_'))==2:
                        img = cv.imread(file_path + '\\' + file)
                        h,w = img.shape[:2]
                        img = cv.resize(img,(SIZE,SIZE))
                        h_rate = h/SIZE
                        w_rate = w/SIZE
                        print(img.shape)
                        img = img/255              #归一化
                        pic.append(img)
                        pts_path = file_path + '\\' + file.split('.')[0]+'.pts'
                        temp = readLmk(pts_path)
                        temp = np.array(temp)
                        for i in range (temp.shape[0]):
                            temp[i][0] = temp[i][0]/w_rate
                            temp[i][1] = temp[i][1]/h_rate
                        pts.append(temp)
                        print("load "+file)
                        print()
                    else:
                        continue
                else:
                    continue
        return pic,pts

train_file_path = 'C:\DataSet\ibug_300W_large_face_landmark_dataset\helen\\trainset'
test_file_path = 'C:\DataSet\ibug_300W_large_face_landmark_dataset\helen\\testset'
x_train_savepath = 'x_train_save.npy'
y_train_savepath = 'y_train_save.npy'
x_test_savepath = 'x_test_save.npy'
y_test_savepath = 'y_test_save.npy'
if __name__ == '__main__':
    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(y_test_savepath) and os.path.exists(y_test_savepath):
        print("---------------------Load datasets---------------------")
        x_train = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
    else:
        print("---------------------Generate datasets---------------------")
        x_train,y_train = generator(train_file_path)
        x_test,y_test = generator(test_file_path)

        print("---------------------Save datasets---------------------")
        np.save(x_train_savepath,x_train)
        np.save(y_train_savepath,y_train)
        np.save(x_test_savepath,x_test)
        np.save(y_test_savepath,y_test)


    img = x_train[90]
    pts = y_train[90]
    print(img.shape)
    print(pts)
    for i in range(len(pts)):
        cv.circle(img,(pts[i][0],pts[i][1]),1,(255,0,0))
    cv.imshow("a",img)
    cv.waitKey()
