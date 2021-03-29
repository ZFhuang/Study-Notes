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
  - [DRCN(2016) 深度递归残差神经网络](#drcn2016-深度递归残差神经网络)
    - [DRCN网络结构](#drcn网络结构)
    - [DRCN简单实现](#drcn简单实现)
    - [DRCN一些经验](#drcn一些经验)
  - [RED(2016) 编码-解码残差](#red2016-编码-解码残差)
    - [RED网络结构](#red网络结构)
    - [RED简单实现](#red简单实现)
    - [RED一些经验](#red一些经验)
  - [DRNN 改进的深度递归残差网络](#drnn-改进的深度递归残差网络)
    - [DRNN网络结构](#drnn网络结构)
    - [DRNN简单实现](#drnn简单实现)
    - [DRNN一些经验](#drnn一些经验)
  - [LapSRN 递归拉普拉斯金字塔](#lapsrn-递归拉普拉斯金字塔)
    - [LapSRN网络结构](#lapsrn网络结构)
    - [LapSRN简单实现](#lapsrn简单实现)
    - [LapSRN一些经验](#lapsrn一些经验)

[从SRCNN到EDSR，总结深度学习端到端超分辨率方法发展历程](https://zhuanlan.zhihu.com/p/31664818)

## SRCNN(2014) 最基础的卷积神经网络

Learning a Deep Convolutional Network for Image Super-Resolution

### SRCNN网络结构

![picture 1](Media/4740a16039a2cac2d91363372190bc756add50cb52fe460c6ef40febe2177f42.png)  

作为最早的超分辨率神经网络, 结构很简单, 就是三个卷积层, 两个激活层的组合, 效果自然也不敢恭维

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

- 多通道超分辨率训练难度大且效果不佳, 因此通过将RGB图像转到YCrCb空间中, 然后只取其Y通道进行超分辨率计算, 完成计算后再配合简单插值处理的CrCb通道. 这种处理方法也被用在了后来的很多超分辨率网络中
- 尽管卷积网络不好训练, 原文使用了ImageNet这样庞大的数据集, 但事实上对于这样很浅的网络用T91就可以得到训练效果
- 用阶段改变学习率的动量SGD效果比Adam更好
- 小batch收敛起来更有效些

## FSRCNN(2016) 更快的SRCNN

Accelerating the Super-Resolution Convolutional Neural Network

### FSRCNN网络结构

![picture 1](Media/f86ebc8d6d46cdeb2bce86d814e8ff1db3ae343c5903d11facb4c58ec61b8c32.png)  

从上面论文中的对比图可以发现其与SRCNN最大的区别就是结尾使用的反卷积层, 反卷积让我们可以直接用没有插值的低分辨率图片进行超分辨率学习, 从而减少超分辨途中的参数数量, 加快网络效率. 并且使用了PReLU作为激活层, 使得激活层本身也可以被学习来提高网络效果

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
- 这个网络训练很快效果也很不错, 思路也有很大参考价值

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

## DRCN(2016) 深度递归残差神经网络

Deeply-Recursive Convolutional Network for Image Super-Resolution

### DRCN网络结构

![picture 1](Media/fdc7917195fda67af1bf5f3a44ad49a50dd75d6989d96662b20dfc314a906c40.png)  

DRCN的亮点就在于中间的递归结构, 其使得每层都按照相同的参数进行了一次处理, 得到的残差通过跳接层相加得到一份结果, 然后所有级数的结果加权合在一起得到最终图像. 由于中间的递归结构每层使用的滤波都是相同的参数, 因此网络的训练难度低了很多, 训练比较高效而且效果也很不错.

### DRCN简单实现

```python
class DRCN(nn.Module):
    def __init__(self, recur_time=16):
        super(DRCN, self).__init__()
        self.recur_time = recur_time
        self.Embedding = nn.Sequential(
            nn.Conv2d(1, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.Inference = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True)
        )
        self.Reconstruction = nn.Sequential(
            nn.Conv2d(256, 1, 3, padding=1),
            nn.ReLU(True)
        )
        self.WeightSum = nn.Conv2d(recur_time, 1, 1)

    def forward(self, img):
        skip = img
        img = self.Embedding(img)
        output = torch.empty(
            (img.shape[0], self.recur_time, img.shape[2], img.shape[3]),device='cuda')
        # 残差连接, 权值共享
        for i in range(self.recur_time):
            img = self.Inference(img)
            output[:, i, :, :] = (skip+self.Reconstruction(img)).squeeze(1)
        # 加权合并
        output = self.WeightSum(output)
        return output
```

### DRCN一些经验

- 论文用到了自适应衰减的学习率和提前终止机制, 这能让训练效率大大提升, pytorch中对应的学习率调整器是: torch.optim.lr_scheduler.ReduceLROnPlateau
- 显存足够的话用大batch大学习率也能得到很好的效果, 还能加快训练
- 论文中提到了越深的递归效果越好, 实践中10层左右的递归就已经能有很好的结果了
- DRCN还用到了称作递归监督的组合损失, 一边计算每个递归层输出的损失一边评判最后的加权损失, 以求所有递归都能得到较好的训练, 目的是避免梯度爆炸/消失. 这是个值得一试的思想, 能让网络一开始更好收敛, 不过直接使用最后的加权误差来进行训练效果也不错.

## RED(2016) 编码-解码残差

Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections

### RED网络结构

![picture 1](Media/d04547cbd6aa1b643133b32973b401dd1e783170c2e6efdca48f62f9c21bad38.png)  

可以看作FSRCNN和VDSR的结合体, 网络自身是卷积和反卷积组合成的对称结构, 每step层就进行一次残差连接, 通过这样反复的特征提取以期望得到质量更高的低分辨率图, 最后用一个反卷积恢复大小. 对称的特征提取组合有学习意义, 可惜最后的反卷积层过于粗暴使得效果不佳

### RED简单实现

```python
class RED(nn.Module):
    def __init__(self,ratio=2, num_feature=32, num_con_decon_mod=5, filter_size=3, skip_step=2):
        super(RED, self).__init__()
        if num_con_decon_mod//skip_step < 1:
            print('Size ERROR!')
            return
        self.num_con_decon_mod = num_con_decon_mod
        self.skip_step = skip_step
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, num_feature, 3, padding=1),
            nn.ReLU(True)
            )
        # 提取特征, 要保持大小不变
        conv_seq = []
        for i in range(0, num_con_decon_mod):
            conv_seq.append(nn.Sequential(
                nn.Conv2d(num_feature, num_feature,
                          filter_size, padding=filter_size//2),
                nn.ReLU(True)
            ))
        self.convs= nn.Sequential(*conv_seq)
        # 反卷积返还特征, 要保持大小不变
        deconv_seq = []
        for i in range(0, num_con_decon_mod):
            deconv_seq.append(nn.Sequential(
                nn.ConvTranspose2d(num_feature, num_feature, filter_size,padding=filter_size//2),
                nn.ReLU(True)
            ))
        self.deconvs=nn.Sequential(*deconv_seq)
        # 真正的放大步骤
        self.output_conv = nn.ConvTranspose2d(num_feature, 1, 3,stride=ratio,padding=filter_size//2)

    def forward(self, img):
        img = self.input_conv(img)
        skips = []
        # 对称残差连接
        for i in range(0, self.num_con_decon_mod):
            if i%self.skip_step==0:
                skips.append(img)
            img = self.convs[i](img)
        for i in range(0, self.num_con_decon_mod):
            img = self.deconvs[i](img)
            if i%self.skip_step==0:
                img=img+skips.pop()
                # 测试中这里不激活效果更好
                # img=torch.relu(img+skips.pop())
        img=self.output_conv(img)
        return img

```

### RED一些经验

- 论文的原始结构中每次残差合并后需要进行一次relu激活, 实践中发现没有这个激活效果也足够
- 嵌套残差连接的写法要记住

## DRNN 改进的深度递归残差网络

### DRNN网络结构

![picture 1](Media/533eab9b3daf0aa9d044826fe6f8084eec0a063b64cefafbabf0e50ba8339f04.png)  

可以理解为VDSR和DRCN的结合体, 网络的大骨架是VSDR式的全残差型, 但是图中每个RB块都是多个递归的小残差块组成, 也就是右图中的conv-conv组合是参数共享的递归形式的, 这样的结构使得不断学习之下也能较好表征深度特征. 但是和其它深度网络类似, 由于参数数目过多导致训练起来非常耗时, 而且要注意在训练函数中进行梯度裁剪减少梯度爆炸或梯度弥散的出现.

### DRNN简单实现

```python
class DRRN(nn.Module):
    # 文中建议的各层组合是令 d=(1+2*num_resid_units)*num_recur_blocks+1 为 20左右
    def __init__(self, num_recur_blocks=1, num_resid_units=15, num_filter=128, filter_size=3):
        super(DRRN, self).__init__()
        # 多个递归块连接
        seq = []
        for i in range(num_recur_blocks):
            if i == 0:
                # 第一个递归块
                seq.append(RecursiveBlock(
                    num_resid_units, 1, num_filter, filter_size))
            else:
                seq.append(RecursiveBlock(num_resid_units,
                                          num_filter, num_filter, filter_size))
        self.residual_blocks = nn.Sequential(*seq)
        # 最终的出口卷积
        self.last_conv = nn.Conv2d(
            num_filter, 1, filter_size, padding=filter_size//2)

    def forward(self, img):
        skip = img
        img = self.residual_blocks(img)
        img = self.last_conv(img)
        # 总残差
        img = skip+img
        return img


class RecursiveBlock(nn.Module):
    # 类似DRCN的递归残差结构, 在RecursiveBlock内部的多个ResidualBlock权值是共享的
    def __init__(self, num_resid_units=3, input_channel=128, output_channel=128, filter_size=3):
        super(RecursiveBlock, self).__init__()
        self.num_resid_units = num_resid_units
        # 递归块的入口卷积
        self.input_conv = nn.Conv2d(
            input_channel, output_channel, filter_size, padding=filter_size//2)
        self.residual_unit = nn.Sequential(
            # 两个conv组, 都有一套激活和加权
            nn.Conv2d(output_channel, output_channel, filter_size,
                      padding=filter_size//2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 1),

            nn.Conv2d(output_channel, output_channel, filter_size,
                      padding=filter_size//2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(True),
            nn.Conv2d(output_channel, output_channel, 1)
        )

    def forward(self, x_b):
        x_b = self.input_conv(x_b)
        skip = x_b
        # 多次残差, 重复利用同一个递归块
        for i in range(self.num_resid_units):
            x_b = self.residual_unit(x_b)
            x_b = skip+x_b
        return x_b
```

### DRNN一些经验

- 深度学习非常难训练, 如果想要短时间出效果的话务必使用较少的递归块并减少滤波器通道数来减少参数数量, 然后通过增加递归次数来弥补性能损失, 例如可以采用B1U5F32的组合
- 网络上有些实现将batchnorm层删去, 但实验证明保留着batchnorm层也能得到效果
- 论文中提到减少递归块的数目, 增加递归次数能够得到好的效果, 论文最后使用了B1U25F128的组合, 两块TITANX在Mix291上训练了4天, 非常耗时, 而且结果仅仅比DRCN好一点, 这个结构很值得怀疑
  
## LapSRN 递归拉普拉斯金字塔

### LapSRN网络结构

![picture 1](Media/331740dc96364098a85c6285e5bde7a61a4fa8fe010868d985ef740b3fb23876.png)  

借用[【超分辨率】Laplacian Pyramid Networks（LapSRN）](https://blog.csdn.net/shwan_ma/article/details/78690974)的网络配图, 可以看到LapSRN的特点是递归结构减少参数量和逐级上采样的结构, 这使得即使面对高倍率的放大任务这个网络也能得到比较稳定的重建结果. 

且由于使用了递归结构和逐级的参数共享, 残差与原始图分流的金字塔结构, 这个网络执行高效, 训练也不困难, 很值得学习.

![picture 2](Media/aacb1f11c072d72eeee36dacef0bb3479bc8c94b6d3830ec431b275acb870f48.png)  

网络还采用了Charbonnier损失函数, 称这个L1loss的变种可以让重建出来的图片不像MSEloss那么模糊, 测试中感觉实际效果有限.

### LapSRN简单实现

```python
class LapSRN(nn.Module):
    def __init__(self, fea_chan=64, scale=2, conv_num=3):
        super(LapSRN, self).__init__()
        # 根据所需的放大倍数计算递归次数
        self.level_num = int(scale/2)
        # 名字带有share的层会在递归中共享参数
        self.share_ski_upsample = nn.ConvTranspose2d(
            1, 1, 4, stride=scale, padding=1)
        self.input_conv = nn.Conv2d(1, fea_chan, 3, padding=1)
        seq = []
        for _ in range(conv_num):
            seq.append(nn.Conv2d(fea_chan, fea_chan, 3, padding=1))
            seq.append(nn.LeakyReLU(0.2, True))
        self.share_embedding = nn.Sequential(*seq)
        self.share_fea_upsample = nn.ConvTranspose2d(
            fea_chan, fea_chan, 4, stride=scale, padding=1)
        self.share_output_conv = nn.Conv2d(fea_chan, 1, 3, padding=1)

    def forward(self, img):
        # 特点是既要准备一个向深层传递的残差层, 也要保持一个向下传递的原始层
        tmp = self.input_conv(img)
        for _ in range(self.level_num):
            skip = self.share_ski_upsample(img)
            img = self.share_embedding(tmp)
            img = self.share_fea_upsample(img)
            tmp = img
            img = self.share_output_conv(img)
            img = img+skip
        return img
```

### LapSRN一些经验

- 反卷积层别忘了要进行双线性参数化
- 实际使用中对于较小的放大倍率很多的卷积层(3-5层足矣)就能得到很好的效果