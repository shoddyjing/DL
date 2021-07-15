#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#朴素感知机：阶跃函数作为激活函数；  单层：与门、与非门、或门；多层(multi-layer perceptron)：异或门。
"""
#theta=-b
def AND(x1,x2):  
   w1, w2, theta = 0.5, 0.5, 0.7
    y=x1*w1+x2*w2
    if y<=theta:
        return 0
    elif y>theta:
        return 1
"""

def AND(x1,x2):  
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    y=np.sum(w*x)+b 
    if y<=0:
        return 0
    elif y>0:
        return 1

def NAND(x1,x2):  
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    y=np.sum(w*x)+b 
    if y<=0:
        return 0
    elif y>0:
        return 1

def OR(x1,x2):  
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    y=np.sum(w*x)+b 
    if y<=0:
        return 0
    elif y>0:
        return 1

#异或门
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y
    
print("x=[0,0]时，与门AND GATE:{}，与非门NAND GATE:{}，或门NAND GATE:{}，异或门XOR GATE:{}".      format(AND(0,0),NAND(0,0),OR(0,0),XOR(0,0)))
print("x=[0,1]时，与门AND GATE:{}，与非门NAND GATE:{}，或门NAND GATE:{}，异或门XOR GATE:{}".      format(AND(0,1),NAND(0,1),OR(0,1),XOR(0,1)))
print("x=[1,1]时，与门AND GATE:{}，与非门NAND GATE:{}，或门NAND GATE:{}，异或门XOR GATE:{}".      format(AND(1,1),NAND(1,1),OR(1,1),XOR(1,1)))

try:    
    get_ipython().system('jupyter nbconvert --to python perceptron.ipynb')
    # python即转化为.py，script即转化为.html
except:
    pass


# In[ ]:





# In[ ]:




