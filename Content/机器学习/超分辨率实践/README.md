# 超分辨率实践

- [超分辨率实践](#超分辨率实践)
  - [SRCNN(2014) 最基础的卷积神经网络](#srcnn2014-最基础的卷积神经网络)
    - [SRCNN网络结构](#srcnn网络结构)
    - [SRCNN简单实现](#srcnn简单实现)
    - [SRCNN一些经验](#srcnn一些经验)
  - [VDSR(2016) 深度残差神经网络](#vdsr2016-深度残差神经网络)
    - [VDSR网络结构](#vdsr网络结构)
    - [VDSR简单实现](#vdsr简单实现)
    - [VDSR一些经验](#vdsr一些经验)

[从SRCNN到EDSR，总结深度学习端到端超分辨率方法发展历程](https://zhuanlan.zhihu.com/p/31664818)

## SRCNN(2014) 最基础的卷积神经网络

Learning a Deep Convolutional Network for Image Super-Resolution

### SRCNN网络结构

![picture 1](Media/4740a16039a2cac2d91363372190bc756add50cb52fe460c6ef40febe2177f42.png)  

结构很简单, 就是三个卷积层, 两个激活层的组合.

### SRCNN简单实现

```python
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 输出大小计算: O=（I-K+2P）/S+1
        # 三层的大小都是不变的, 通道数在改变
        # 原文没有使用padding因此图片会变小, 这里使用了padding
        self.conv1=nn.Conv2d(1,64,9, padding=4)
        self.conv2=nn.Conv2d(64,32,1, padding=0)
        self.conv3=nn.Conv2d(32,1,5, padding=2)
    
    def forward(self, img):
        # 三层的学习率不同
        # 两个激活层
        img=torch.relu(self.conv1(img))
        img=torch.relu(self.conv2(img))
        # 注意最后一层不要激活
        return self.conv3(img)
```

### SRCNN一些经验

- 多通道超分辨率训练难度大且效果不佳, 因此通过将RGB图像转到YCrCb空间中, 然后只取其Y通道进行超分辨率计算, 完成计算后再配合简单插值处理的CrCb通道
- 卷积网络不好训练, 原文使用了ImageNet这样庞大的数据集, 事实上T91就可以得到训练效果
- 用阶段改变学习率的动量SGD效果比Adam更好
- 小batch收敛起来更有效些

## VDSR(2016) 深度残差神经网络

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

### VDSR网络结构

![picture 1](Media/36264453afa07c6f10205f029e277b59bffb4dc32f1d09522daa2c3280055840.png)  

使用大量3*3的 卷积-激活块 进行串联, 用padding保证输入输出的尺寸, 整体用一个残差来优化训练. 网络的目标是得到的残差尽可能接近HR-LR, 用MSE作为loss训练. 

### VDSR简单实现

```python
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.body=VDSR_Block()
    
    def forward(self, img):
        img=self.body(img)
        for i in range(img.shape[0]):
            # 由于Relu的存在, 得到的残差需要移动平均来应用
            img[i,0,:,:]-=torch.mean(img[i,0,:,:])
        # 由于网络只有一个残差块, 所以把残差的相加写到了loss计算中
        return img
        
class VDSR_Block(nn.Module):
    def __init__(self):
        super(VDSR_Block, self).__init__()
        self.inp=nn.Conv2d(1,64,3,bias=False,padding=1)
        seq=[]
        # 20层卷积
        for j in range(20):
            seq.append(nn.Conv2d(64,64,3,padding=1))
            seq.append(nn.ReLU(True))
        self.conv=nn.Sequential(*seq)
        self.out=nn.Conv2d(64,1,3,padding=1)
    
    def forward(self, img):
        img=torch.relu(self.inp(img))
        img=self.conv(img)
        img=self.out(img)
        return img
```

### VDSR一些经验

- 很不好训练, 效果也不理想, 不知道是不是实现有问题, 也可能是只有一个残差块的缺点, 梯度很快消失
- 用阶段改变学习率的动量SGD来训练