import cv2 #opencv库，用于读取图片等操作
import os
import numpy as np 
import pandas as pd  
from sklearn.preprocessing import StandardScaler #标准差标准化
from sklearn.svm import SVC                      #svm包中SVC用于分类
from sklearn.decomposition import PCA            #特征分解模块的PCA用于降维
def get_data(x,y):
    file_path='./faces_4' #设置文件路径（这里为当前目录下的ORL文件夹）
    targets = os.listdir(file_path)
    train_set = np.zeros(shape=[1,112*92])  #train_set用于获取的数据集
    train_set = pd.DataFrame(train_set)     #将train_set转换成DataFrame类型
    target=[]  
    i = 0
    for each in targets:
        cur_path = file_path + '/' + each + '/'
        i = i + 1
        img_names = os.listdir(cur_path)
        for j in range(x,y):                #j用于遍历每个文件夹的对应的x到y-1的图片
            target.append(i)    
            img_name = img_names[j]            #读入标签（图片文件夹中的人脸是同一个人的）
            img = cv2.imread(cur_path + str(img_name), cv2.IMREAD_GRAYSCALE) #读取图片，第二个参数表示以灰度图像读入
            img=img.reshape(1,img.shape[0]*img.shape[1]) #将读入的图片数据转换成一维
            img=pd.DataFrame(img)           #将一维的图片数据转成DataFrame类型
            train_set=pd.concat([train_set,img],axis=0)#按行拼接DataFrame矩阵                            #标签列表
    
    train_set.index=list(range(0,train_set.shape[0])) #设置 train_set的行索引
    train_set.drop(labels=0,axis=0,inplace=True) #删除行索引为0的行（删除第一行）
    target=pd.DataFrame(target)             #将标签列表转成DataFrame类型
    return train_set,target                 #返回数据集和标签

if __name__ == '__main__':
    #1、获取数据
    face_data_train,face_target_train=get_data(0,15) #读取前五张图片为训练集
    face_data_test,face_target_test=get_data(16,20)  #读取后五张图片为测试集
    #2、数据标准化 标准差标准化
    stdScaler = StandardScaler().fit(face_data_train) 
    face_trainStd = stdScaler.transform(face_data_train)
    #stdScaler = StandardScaler().fit(face_data_test) 
    face_testStd = stdScaler.transform(face_data_test)
    #3、PCA降维
    pca = PCA(n_components=20).fit(face_trainStd) 
    face_trainStd = pca.transform(face_trainStd) 
    face_testStd = pca.transform(face_testStd)
    #4、建立SVM模型  默认为径向基核函数kernel='rbf' 多项式核函数kernel='poly'
    #svm = SVC().fit(face_trainStd,face_target_train)
    svm = SVC(kernel='poly').fit(face_trainStd,face_target_train)
    print('建立的SVM模型为：\n',svm)
    #4、预测训练集结果
    face_target_pred = svm.predict(face_testStd)
    print('预测前10个结果为：\n',face_target_pred[:10])
    face_target_test=face_target_test.values    #Dataframe转ndarray方便后面准确率的判断
    true=0
    ## 求出预测和真实一样的数目
    #true = np.sum(face_target_pred == face_target_test )
    for i in range(0,200):
        if face_target_pred[i] == face_target_test[i]:
            true=true+1
    print('预测对的结果数目为：', true)
    print('预测错的的结果数目为：', face_target_test.shape[0]-true)
    print('预测结果准确率为：', true/face_target_test.shape[0])
