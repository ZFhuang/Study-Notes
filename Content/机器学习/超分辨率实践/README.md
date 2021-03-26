# 超分辨率实践

- [超分辨率实践](#超分辨率实践)
  - [SRCNN(2014) 最基础的卷积神经网络](#srcnn2014-最基础的卷积神经网络)
    - [SRCNN网络结构](#srcnn网络结构)
    - [SRCNN简单实现](#srcnn简单实现)
    - [SRCNN一些经验](#srcnn一些经验)
  - [FSRCNN(2016) 更快的SRCNN](#fsrcnn2016-更快的srcnn)
    - [FSRCNN网络结构](#fsrcnn网络结构)
    - [FSRCNN简单实现](#fsrcnn简单实现)
    - [FSRCNN一些经验](#fsrcnn一些经验)
  - [ESPCN(2016) 实时进行的亚像素卷积](#espcn2016-实时进行的亚像素卷积)
    - [ESPCN网络结构](#espcn网络结构)
    - [ESPCN的简单实现](#espcn的简单实现)
    - [ESPCN一些经验](#espcn一些经验)
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

## FSRCNN(2016) 更快的SRCNN

Accelerating the Super-Resolution Convolutional Neural Network

### FSRCNN网络结构

![picture 1](Media/f86ebc8d6d46cdeb2bce86d814e8ff1db3ae343c5903d11facb4c58ec61b8c32.png)  

其与SRCNN最大的区别就是结尾使用的反卷积层, 通过反卷积让我们可以直接用没有插值的低分辨率图片进行超分辨率学习, 从而减少超分辨途中的参数数量, 加快网络效率. 并且使用了PReLU作为激活层, 使得激活层本身也可以被学习来提高网络效果

### FSRCNN简单实现

```python
class FSRCNN(nn.Module):
    def __init__(self,d,s,m,ratio=2):
        super(FSRCNN, self).__init__()
        feature_extraction=nn.Conv2d(1,d,5, padding=2)
        shrinking=nn.Conv2d(d,s,1)
        seq=[]
        for i in range(m):
            seq.append(nn.Conv2d(s,s,3,padding=1))
        non_linear=nn.Sequential(*seq)
        expanding=nn.Conv2d(s,d,1,padding=0)
        # 反卷积尺寸计算 O=(I-1)×s+k-2P 
        deconvolution=nn.ConvTranspose2d(d,1,9,stride=ratio,padding=4)
        self.body=nn.Sequential(
            feature_extraction,
            nn.PReLU(),
            shrinking,
            nn.PReLU(),
            non_linear,
            nn.PReLU(),
            expanding,
            nn.PReLU(),
            deconvolution
        )
    
    def forward(self, img):
        return self.body(img)
```

### FSRCNN一些经验

- 由于输入输出大小不一样, 因此误差计算等需要注意写好
- 由于反卷积的计算问题, 实际输出的结果图会比HR图小一个像素, 因此在使用的时候需要将HR层手动裁剪一个像素来适配网络

## ESPCN(2016) 实时进行的亚像素卷积

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

### ESPCN网络结构

![picture 1](Media/e2ad2f7b84a56f4f48429f22ff9089accda61c4eee961f09bb2d34687dc047dd.png)  

核心的优化在于最后一层的亚像素卷积过程, 其思想就是将卷积得到的多通道低分辨率图的像素按照周期排列得到高分辨率的图片, 这样训练出能够共同作用来增强分辨率的多个滤波器. 借用[一边Upsample一边Convolve：Efficient Sub-pixel-convolutional-layers详解](https://oldpan.me/archives/upsample-convolve-efficient-sub-pixel-convolutional-layers)的示意图可以更好理解亚像素卷积的过程.

![picture 3](Media/3913f4b4d67b37cb06b6ddd12d71841fb250715ddf04ac6e467ddbb0ef59a5b7.png)  

### ESPCN的简单实现

```python
class ESPCN(nn.Module):
    def __init__(self, ratio=2):
        super(ESPCN, self).__init__()
        self.add_module('n1 conv', nn.Conv2d(1,64,5,padding=2))
        self.add_module('tanh 1',nn.Tanh())
        self.add_module('n2 conv', nn.Conv2d(64,32,3,padding=1))
        self.add_module('tanh 2',nn.Tanh())
        self.add_module('n3 conv', nn.Conv2d(32,1*ratio*ratio,3,padding=1))
        # 亚像素卷积
        self.add_module('pixel shuf',nn.PixelShuffle(ratio))
    
    def forward(self, img):
        for module in self._modules.values():
            img = module(img)
        return img
```

### ESPCN一些经验

- ESPCN用小LR块训练效果更好
- 注意网络最后一层不要再放入激活层了, 会有反作用

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