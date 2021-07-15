#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ch04.02 page 85
import sys, os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d

sys.path.append(os.pardir) #将父目录（上一级）os.pardir添加到系统搜索路径
#print(os.pardir) # 输出‘..’
from dataset.mnist import load_mnist #在上一级中搜索dataset文件夹（模块）并得到其中mnist.py文件
from PIL import Image


def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))  #unit8, 0~255内存储图像的np数组
    pil_img.show()

#step function
def step_function(x):# if x>=0:  return 1 elif x<0: return 0
    #y=np.array(x>0)
    y=np.array(x>0, dtype=np.int)         
    return y

def sigmoid(x):
    y=1/(1+np.exp(-x))  #和非数组计算的广播功能
    return y

#Rectified Linear Unit function
def ReLU(x):
  #  y=np.array(x)
  #  y[y<=0]=0.0
  #  return y
    return(np.maximum(x,0)) 

#------------------------输出层函数------------------------------
#1. identity_function 恒等函数，apply to regression problems 适合回归问题
def identity(a):
    return a

#2. softmax_function ，apply to classification problems 适合分类问题
def softmax(a):
    if len(a.shape)>1:
        c= np.max(a,axis=1) # 得到每行的最大值，用于缩放每行的元素，避免溢出
        a -= c.reshape((a.shape[0],1)) # 利用性质缩放元素
        a = np.exp(a) # 计算所有值的指数
        c = np.sum(a, axis = 1) # 每行求和        
        a /= c.reshape((a.shape[0], 1)) # 求softmax
    else:
        c = np.max(a) # 得到最大值
         a-= c # 利用最大值缩放数据
        a = np.exp(a) # 对所有元素求指数        
        c = np.sum(c) # 求元素和
        a /= c # 求somftmax
        #c=np.max(a,axis=1)
        #exp_a=np.exp(a-c)    #通过减去最大值防止因e指数后各个数值相差太大而出现的溢出现象。
        #sum_exp_a=np.sum(exp_a)
        #y=exp_a/sum_exp_a  #sum_exp_a为一个常数，此处使用了广播功能
        #return y
    return a           #sum(y)=1,且y的大小顺序同输入a(y=e**a为单调递增函数)，所以softmax层有时会被省略
#-----------------------损失函数-------------------------------
#1.mean-squared error 均方误差
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

#2.cross entropy error 交叉熵误差
def cross_entropy_error_origin(y,t):
    delta=1e-7 #防止np.log(0)的出现导致-inf（-infinite，无限小），计算无法继续
    return -np.sum(t*np.log(y+delta)) #此值仅由对应正确标签位置的y决定，此时y越大，则说明预测更准，误差越小

#3.mini-batch cross entropy error 批处理交叉熵误差
def cross_entropy_error(y,t):
    delta=1e-7 #防止np.log(0)的出现导致-inf（-infinite，无限小），计算无法继续
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+delta))/batch_size #此值仅由对应正确标签位置的y决定，此时y越大，则说明预测更准，误差越小


def numerical_diff(f,x): #求单个变量的中央差，是导数的近似
    h=1e-4                    
    return (f(x+h)-f(x-h))/(2*h) 

def numerical_gradient_one(f,x): #求一维梯度，即所有变量的偏导数
    h=1e-4
    grad=np.zeros_like(x)
    for i in range(x.size):
        fx1=x[i]+h
        fx2=x[i]-h
        grad[i]=(f(fx1)-f(fx2))/(2*h)
    return grad

def numerical_gradient(f,x):
    if x.ndim==1:
        return numerical_gradient_one(f,x)
    else:
        grad=np.zeros_like(x)
        for i,xx in enumerate(x):
            grad[i]=numerical_gradient_one(f,xx)
        return grad

def init_network(input_size,hidden_size1,hidden_size2,output_size): 
    network={}    #创建字典型变量存储权重和偏置值
    network['w1']=np.random.randn(input_size,hidden_size1)*2**0.5/np.sqrt(input_size) #高斯分布+He初始值
    network['b1']=np.zeros(hidden_size1)
    network['w2']=np.random.randn(hidden_size1,hidden_size2)*2**0.5/np.sqrt(hidden_size1)
    network['b2']=np.zeros(hidden_size2)
    network['w3']=np.random.randn(hidden_size2,output_size)*2**0.5/np.sqrt(hidden_size2)
    network['b3']=np.array(output_size)
    return network

def forward(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)     #中间层值为a->z，输出层为y
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax(a3) #softmax适合分类问题
    #y=identity(a3)
    return y
'''
def get_data():
    (x_train, t_train), (x_test, t_test)= load_mnist(flatten=True, normalize=True, one_hot_label=True)
   # print("shape of train data:{}".format(x_train.shape))
    print("shape of train label:{}".format(t_train.shape))
   # print("shape of test data:{}".format(x_test.shape))
    print("shape of test label:{}".format(t_test.shape))
    return x_train, t_train, x_test, t_test
'''
    
def write_txt_data(fname, wtype, data0):
    f=open(fname,wtype)
    for i in range(len(data0)):
        data=str(data0[i]).replace('[','').replace(']','')
        data=data.replace("'","").replace(',','')+'\n'
        f.write(data)
    f.close()
    print('{}文件保存完毕!'.format(fname))


try:    
    get_ipython().system('jupyter nbconvert --to python functions_methods.ipynb')
    # python即转化为.py，script即转化为.html
except:
    pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




