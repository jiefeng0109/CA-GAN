# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:39:40 2017

@author: 49603
"""


import scipy.io as sio #引用包，进行.mat文件转换等
import numpy as np#引用numpy包 矩阵计算
import sys#该模块提供了对解释器使用或维护的一些对象的访问，以及与解释器进行强烈交互的函数
from sklearn.decomposition import PCA#矩阵分解算法，PCA、NMF或ICA。该模块的大多数算法都是降维
from sklearn import preprocessing#预处理包提供了几个常用的实用程序函数和transformer类，可以将原始特征向量转换为更适合于后面评估器的表示形式
import math#提供了对C标准定义的数学函数的访问
from cell import ConvLSTMCell
import os

os.environ["CUDA_VISIBLE_DEVICES"]='1'
'''class_number=16#分类类别16类'''
class_number=10#分类类别16类
neighbor_size_thd=27#邻域高27
neighbor_size_twd=27#邻域宽27

def contrary_one_hot(label):#给出数据标签
    size=len(label)
    label_ori=np.empty(size)#定义元组形式【行*列】，默认存为实数
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1#返回沿轴【化为一行/列】的最大值的索引，进第一个最大值的索引
    return label_ori#通过标签最大值给出所属类别

def get_data(flag):#获取数据
    #hsi=r'/home/fengjie/G/data/Indian_pines/Indian_pines_corrected.mat'#真实数据
    #gnd=r'/home/fengjie/G/data/Indian_pines/Indian_pines_gt.mat'#真实类标
    hsi='G:\data\PaviaU\PaviaU.mat'
    gnd='G:\data\PaviaU\PaviaU_gt.mat'
    dic=sio.loadmat(hsi)
    dicgd=sio.loadmat(gnd)#加载MATLAB文件,dic是个字典

    total_data=dic['paviaU']
    total_label=dicgd['paviaU_gt']
   
    
    data,data_unl,label,extracted_pixel_ind,extracted_unlpixel_ind=pre_data(total_data,total_label,neighbor_size_twd,neighbor_size_thd,flag=flag,n_principle=20)#调用下面函数
    train_data,test_data,train_label,test_label,train_number,test_number,class_mean=process_data(data,label)
    #train_data,test_data,train_label,test_label,train_number,test_number=process_data(data,label)
    
    return train_data,train_label,train_number,test_data,test_label,class_mean,data,extracted_pixel_ind,extracted_unlpixel_ind
    #return train_data,train_label,train_number,test_data,test_label,data,extracted_pixel_ind,extracted_unlpixel_ind
    
    

def pre_data(hsi_img,gnd_img,window_size_twd,window_size_thd,flag,n_principle):
    

    #flag=0,for spectral;flag=1,for spatial,flag=2,for spectral and spatial joint
    if flag==0:#光谱维
       hsi_img=hsi_img.astype('float32')#数据类型转换(本身是数组形式)，将int等变为float形式
       
       width=hsi_img.shape[0]#返回img的第一维长度
       length=hsi_img.shape[1]
       dim=hsi_img.shape[2]#长宽高
       
       reshaped_img = hsi_img.reshape(length*width, dim)#数据平面化为一列，dim深维度
       data_norm = norm(reshaped_img)#归一化
       reshaped_gnd =gnd_img.reshape(gnd_img.size)#将带有标签的数据reshape为特定形状的矩阵
       
       extracted_pixel= (reshaped_gnd > 0)#获取类标>0的像素
       extracted_pixel_unl=(reshaped_gnd==0)#类标为0的像素
       '''extracted_pixel是bool（布尔型）矩阵，可以选择的像素对应位置为True,类别为0的对应位置为False'''
       gndtruth = reshaped_gnd[extracted_pixel]
       '''gndtruth是选择的非零的类标按顺序组成的矩阵'''
       extracted_pixel_ind = np.arange(reshaped_gnd.size)[extracted_pixel]
       #在给定的间隔内返回均匀间隔的值，从0开始，以所得类标非0的像素值为间隔，获得非0标签的像素矩阵
       '''extracted_pixel_ind是在一维向量中类标非零像素对应位置组成的矩阵,以0开头'''
       extracted_unlpixel_ind=np.arange(reshaped_gnd.size)[extracted_pixel_unl]#获得0标签的像素矩阵
       
       data_spectral = np.zeros([extracted_pixel_ind.size, dim], dtype='float32')#对标签非0的像素生成给定形状的0.0数组
       data_spectral_unl=np.zeros([extracted_unlpixel_ind.size,dim],dtype='float32')#对标签为0的像素生成给定形状的0.0数组
       i = 0
       for ipixel in extracted_pixel_ind:
           data_spectral[i,:] = data_norm[ipixel,:]#标签非0数据归一化
           i += 1
       
       j= 0
       for ipixel_unl in extracted_unlpixel_ind:
           data_spectral_unl[j,:] = data_norm[ipixel_unl,:]#标签为0数据归一化
           j += 1
           
       return data_spectral,data_spectral_unl,gndtruth#返回光谱维度标签非0数据，标签为0数据，以及选择的非零的类标按顺序组成的矩阵
          
    elif flag==1:#空间维
        hsi_img=hsi_img.astype('float32')#数据类型转换(本身是数组形式)，将int等变为float形式
        length = hsi_img.shape[0]
        width = hsi_img.shape[1]
        dim = hsi_img.shape[2]
        '''归一化'''
        reshaped_img = hsi_img.reshape(length*width, dim)
        data_norm = norm(reshaped_img)
        '''PCA'''
        pca = PCA(n_components = n_principle)#降维后的特征维度数目
        data_PCA = pca.fit_transform(data_norm)#降维后的数据
        data_PCA = data_PCA.reshape(length,width,n_principle)#降维后的数据形式长宽深（n_principle）即为降维后的维度
        '''填充'''
        data_PCA_expand = np.empty([length+window_size_twd-1,width+window_size_twd-1,n_principle])#返回给定形状和类型的新数组，而不初始化条目(即值随机)，长和宽分别创建27-1个像素点
        threshold = int((window_size_twd-1) / 2)#阈值=13
        for i in range(n_principle):
            data_PCA_expand[:,:,i] = np.lib.pad(data_PCA[:,:,i], ((threshold, threshold), (threshold,threshold)), 'symmetric')#进行镜像扩展，长宽维度前面和后面均扩展13个长度，symmetric表示填充的值为按边缘进行反射所得的值，即镜像填充。
        
        '''reshaped_gnd_expand是21025的向量'''#（145*145=21025）
        reshaped_gnd = gnd_img.reshape(length*width)
        '''取窗'''
        '''extracted_pixel是bool（布尔型）矩阵，可以选择的像素对应位置为True,类别为0的对应位置为False'''
        extracted_pixel = (reshaped_gnd > 0) 
        extracted_pixel_unl=(reshaped_gnd==0)
        gndtruth = reshaped_gnd[extracted_pixel]
        '''gndtruth是选择的非零的类标按顺序组成的矩阵'''
        extracted_pixel_ind = np.arange(reshaped_gnd.size)[extracted_pixel]
        extracted_unlpixel_ind=np.arange(reshaped_gnd.size)[extracted_pixel_unl]
        '''extracted_pixel_ind是在一维向量中类标非零像素对应位置组成的矩阵,以0开头'''
        data_spatial = np.zeros([extracted_pixel_ind.size, window_size_twd, window_size_twd, n_principle], dtype='float32')
        data_spatial_unl=np.zeros([extracted_unlpixel_ind.size,window_size_twd,window_size_twd,n_principle],dtype='float32')#构建空间维度有无标签样本的0矩阵
        i = 0#类标非0，对像素进行计算
        for ipixel in extracted_pixel_ind:
            index_c_ori = ipixel % width#求余？？？
            index_r_ori = int(ipixel / width)#求商？？？
            index_c=index_c_ori+threshold
            index_r=index_r_ori+threshold
            data_spatial[i,:,:,:] = data_PCA_expand[index_r-threshold : index_r+threshold+1, index_c-threshold : index_c+threshold+1,:]
            i += 1
       
        j = 0#类标为0
        for ipixel_unl in extracted_unlpixel_ind:
            index_c_unl_ori = ipixel_unl % width
            index_r_unl_ori = int(ipixel_unl / width)
            index_c=index_c_unl_ori+threshold
            index_r=index_r_unl_ori+threshold
            data_spatial_unl[j,:,:,:] = data_PCA_expand[index_r-threshold : index_r+threshold+1, index_c-threshold : index_c+threshold+1,:]
            j += 1
        # if we want to merge data, merge it
        return data_spatial,data_spatial_unl,gndtruth,extracted_pixel_ind,extracted_unlpixel_ind
        
    elif flag==2:#空谱维
        hsi_img=hsi_img.astype('float32')
        length = hsi_img.shape[0]
        width = hsi_img.shape[1]
        dim = hsi_img.shape[2]

        reshaped_img =hsi_img.reshape(length*width, dim)
        data_norm = norm(reshaped_img)
        
        pca = PCA(n_components=n_principle)
        data_PCA = pca.fit_transform(data_norm)
        data_PCA=data_PCA.reshape(length,width,n_principle)
        '''hsi_img=data_norm.reshape(length,width,dim)'''
        hsi_img_expand=np.empty([length+window_size_thd-1,width+window_size_thd-1,n_principle])
        threshold = int((window_size_thd-1) / 2)
        for i in range(n_principle):
            hsi_img_expand[:,:,i]=np.lib.pad(data_PCA[:,:,i], ((threshold, threshold), (threshold,threshold)), 'symmetric')
        
         
        '''extracted_pixel是bool矩阵，可以选择的像素对应位置为True,类别为0的对应位置为False'''
        reshaped_gnd = gnd_img.reshape(length*width)
        extracted_pixel = (reshaped_gnd > 0) 
        extracted_pixel_unl=(reshaped_gnd==0)  
        gndtruth = reshaped_gnd[extracted_pixel]
        '''gndtruth是选择的非零的类标按顺序组成的矩阵'''
        extracted_pixel_ind = np.arange(reshaped_gnd.size)[extracted_pixel]
        extracted_unlpixel_ind=np.arange(reshaped_gnd.size)[extracted_pixel_unl]
        '''extracted_pixel_ind是在一维向量中类标非零像素对应位置组成的矩阵,以0开头'''
        data_joint = np.zeros([extracted_pixel_ind.size, window_size_thd, window_size_thd, n_principle], dtype='float32')
        data_joint_unl=np.zeros([extracted_unlpixel_ind.size,window_size_thd,window_size_thd,n_principle],dtype='float32')
        i = 0
        for ipixel in extracted_pixel_ind:
            index_c_ori = ipixel % width
            index_r_ori = int(ipixel / width)
            index_c=index_c_ori+threshold
            index_r=index_r_ori+threshold
            data_joint[i,:,:,:] = hsi_img_expand[index_r-threshold : index_r+threshold+1, index_c-threshold : index_c+threshold+1,:]
            i += 1
       
        j= 0
        for ipixel_unl in extracted_unlpixel_ind:
            index_c_unl_ori = ipixel_unl % width
            index_r_unl_ori = int(ipixel_unl / width)
            index_c=index_c_unl_ori+threshold
            index_r=index_r_unl_ori+threshold
            data_joint_unl[j,:,:,:] = hsi_img_expand[index_r-threshold : index_r+threshold+1, index_c-threshold : index_c+threshold+1,:]
            j += 1
        return data_joint,data_joint_unl, gndtruth
    
def process_data(img,label):
    
    count=np.bincount(label)#在非负整数数组中计算每个值的次数。
    
    size=np.shape(img)#返回数组的形状
    width=size[1]
    length=img.shape[2]
    band=img.shape[3]
    
    data_c1=img[np.where(label==1)]#找label中为1的位置，
    
    data_c2=img[np.where(label==2)]
    data_c3=img[np.where(label==3)]
    data_c4=img[np.where(label==4)]
    data_c5=img[np.where(label==5)]
    data_c6=img[np.where(label==6)]
    data_c7=img[np.where(label==7)]
    data_c8=img[np.where(label==8)]
    data_c9=img[np.where(label==9)]
    '''
    data_c10=img[np.where(label==10)]
    data_c11=img[np.where(label==11)]
    data_c12=img[np.where(label==12)]
    data_c13=img[np.where(label==13)]
    data_c14=img[np.where(label==14)]
    data_c15=img[np.where(label==15)]
    data_c16=img[np.where(label==16)]
    '''
    
    trd1,ted1=separate(data_c1)#调用自定义separate函数,返回训练数据和测试数据
    trd2,ted2=separate(data_c2)
    trd3,ted3=separate(data_c3)
    trd4,ted4=separate(data_c4)
    trd5,ted5=separate(data_c5)
    trd6,ted6=separate(data_c6)
    trd7,ted7=separate(data_c7)
    trd8,ted8=separate(data_c8)
    trd9,ted9=separate(data_c9)
    '''
    trd10,ted10=separate(data_c10)
    trd11,ted11=separate(data_c11)
    trd12,ted12=separate(data_c12)
    trd13,ted13=separate(data_c13)
    trd14,ted14=separate(data_c14)
    trd15,ted15=separate(data_c15)
    trd16,ted16=separate(data_c16)
    '''
    

    class_mean=np.empty([class_number,width*length*band])#返回给定形状和类型的新数组，而不初始化条目(即值随机)
    class_mean[0]=np.mean(np.reshape(trd1,[-1,width*length*band]),axis=0)#重塑为指定形状的数组后，沿着指定的轴计算算术平均值，axis=0，按列求平均值
    class_mean[1]=np.mean(np.reshape(trd2,[-1,width*length*band]),axis=0)
    class_mean[2]=np.mean(np.reshape(trd3,[-1,width*length*band]),axis=0)
    class_mean[3]=np.mean(np.reshape(trd4,[-1,width*length*band]),axis=0)
    class_mean[4]=np.mean(np.reshape(trd5,[-1,width*length*band]),axis=0)
    class_mean[5]=np.mean(np.reshape(trd6,[-1,width*length*band]),axis=0)
    class_mean[6]=np.mean(np.reshape(trd7,[-1,width*length*band]),axis=0)
    class_mean[7]=np.mean(np.reshape(trd8,[-1,width*length*band]),axis=0)
    class_mean[8]=np.mean(np.reshape(trd9,[-1,width*length*band]),axis=0)
    '''
    class_mean[9]=np.mean(np.reshape(trd10,[-1,width*length*band]),axis=0)
    class_mean[10]=np.mean(np.reshape(trd11,[-1,width*length*band]),axis=0)
    class_mean[11]=np.mean(np.reshape(trd12,[-1,width*length*band]),axis=0)
    class_mean[12]=np.mean(np.reshape(trd13,[-1,width*length*band]),axis=0)
    class_mean[13]=np.mean(np.reshape(trd14,[-1,width*length*band]),axis=0)
    class_mean[14]=np.mean(np.reshape(trd15,[-1,width*length*band]),axis=0)
    class_mean[15]=np.mean(np.reshape(trd16,[-1,width*length*band]),axis=0)
    '''

 
    print(len(trd1),len(trd2),len(trd3),len(trd4),len(trd5),len(trd6),len(trd7),len(trd8),len(trd9))
    #print(len(trd10),len(trd11),len(trd12),len(trd13),len(trd14),len(trd15),len(trd16))#训练数据长度
    #train_data=np.vstack((trd1,trd2,trd3,trd4,trd5,trd6,trd7,trd8,trd9,trd10,trd11,trd12,trd13,trd14,trd15,trd16))#将数组按顺序垂直排列，一行一个[]
    #test_data=np.vstack((ted1,ted2,ted3,ted4,ted5,ted6,ted7,ted8,ted9,ted10,ted11,ted12,ted13,ted14,ted15,ted16))
    train_data=np.vstack((trd1,trd2,trd3,trd4,trd5,trd6,trd7,trd8,trd9))#将数组按顺序垂直排列，一行一个[]
    test_data=np.vstack((ted1,ted2,ted3,ted4,ted5,ted6,ted7,ted8,ted9))
    
    
    train_number=len(train_data)
    test_number=len(test_data)#赋值
    
    lbr1=one_hot(trd1,1)#针对训练数据返回标签为1的类标
    lbr2=one_hot(trd2,2)
    lbr3=one_hot(trd3,3)
    lbr4=one_hot(trd4,4)
    lbr5=one_hot(trd5,5)
    lbr6=one_hot(trd6,6)
    lbr7=one_hot(trd7,7)
    lbr8=one_hot(trd8,8)
    lbr9=one_hot(trd9,9)
    '''
    lbr10=one_hot(trd10,10)
    lbr11=one_hot(trd11,11)
    lbr12=one_hot(trd12,12)
    lbr13=one_hot(trd13,13)
    lbr14=one_hot(trd14,14)
    lbr15=one_hot(trd15,15)
    lbr16=one_hot(trd16,16)
    '''
    
    lbe1=one_hot(ted1,1)#针对测试数据返回标签为1的类标
    lbe2=one_hot(ted2,2)
    lbe3=one_hot(ted3,3)
    lbe4=one_hot(ted4,4)
    lbe5=one_hot(ted5,5)
    lbe6=one_hot(ted6,6)
    lbe7=one_hot(ted7,7)
    lbe8=one_hot(ted8,8)
    lbe9=one_hot(ted9,9)
    '''
    lbe10=one_hot(ted10,10)
    lbe11=one_hot(ted11,11)
    lbe12=one_hot(ted12,12)
    lbe13=one_hot(ted13,13)
    lbe14=one_hot(ted14,14)
    lbe15=one_hot(ted15,15)
    lbe16=one_hot(ted16,16)
    '''
    
    #train_label=np.vstack((lbr1,lbr2,lbr3,lbr4,lbr5,lbr6,lbr7,lbr8,lbr9,lbr10,lbr11,lbr12,lbr13,lbr14,lbr15,lbr16))
    #test_label=np.vstack((lbe1,lbe2,lbe3,lbe4,lbe5,lbe6,lbe7,lbe8,lbe9,lbe10,lbe11,lbe12,lbe13,lbe14,lbe15,lbe16))#进行类标按一行一个数组[]排列
    train_label=np.vstack((lbr1,lbr2,lbr3,lbr4,lbr5,lbr6,lbr7,lbr8,lbr9))
    test_label=np.vstack((lbe1,lbe2,lbe3,lbe4,lbe5,lbe6,lbe7,lbe8,lbe9))#进行类标按一行一个数组[]排列
    

    perm = np.arange(train_number)#在给定的间隔内[0,train_number]返回均匀间隔的值
    np.random.shuffle(perm)#沿着多维数组的第一个轴旋转数组。子数组的顺序发生了变化，但其内容保持不变。
    train_data = train_data[perm]
    train_label = train_label[perm]

    perm = np.arange(test_number)
    np.random.shuffle(perm)
    test_data = test_data[perm]
    test_label = test_label[perm]

    return train_data,test_data,train_label,test_label,train_number,test_number,class_mean
   
    
def separate(per_class_data):#返回训练和测试数据
    
    per_class_number = len(per_class_data)#每类个数
    order = np.random.permutation(range(per_class_number))#返回一个对象，该对象从start到stop(per_class_number)生成一个整数序列;随机返回一个有限范围序列,如果x是一个多维数组，只沿着它的第一个索引进行拖拽。
    per_class_data=per_class_data[order]
    train_data_number=int(np.round(per_class_number*0.02))#确定训练数据所占比例，用小数(默认的0位)把一个数字圆到一个给定的精度。当调用一个参数时，返回一个int类型，否则与number类型相同
    #train_data_number=int(np.round(34))
    train_data=per_class_data[0:train_data_number]
    test_data=per_class_data[train_data_number:]
    
    return train_data,test_data
    
def norm(pixels):#归一化
          
    #convert
    shape=np.shape(pixels)
    
    for i in range(shape[0]):
        pixels[i,:] = preprocessing.normalize(pixels[i,:].reshape(1,-1),norm='l2')#归一化，（l2注意不是12）其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
        '''minrow=min(pixels[i])
        maxrow=max(pixels[i])
        #for j in range(shape[1]):
        pixels[i]=(pixels[i]-minrow)/(maxrow-minrow)'''
    
    return pixels
    
def one_hot(data,i):#返回对应真实类标标签
   '''one_hot_array=np.zeros([len(arr),column])
   for i in range(len(arr)):
       one_hot_array[i,arr[i]-1]=1
    
   return one_hot_array'''
   number=len(data)
   label=np.zeros([number,class_number])
   for j in range(number):
       label[j][i-1]=1

   return label
def contrary_one_hot(label):
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori