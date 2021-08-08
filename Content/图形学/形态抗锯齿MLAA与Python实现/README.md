# 形态抗锯齿MLAA与Python实现

- [形态抗锯齿MLAA与Python实现](#形态抗锯齿mlaa与python实现)
  - [总览](#总览)
    - [参考资料](#参考资料)
    - [流程概览](#流程概览)
  - [流程](#流程)
    - [查找边缘](#查找边缘)
    - [模式分类](#模式分类)
    - [权重计算](#权重计算)
    - [颜色混合](#颜色混合)
  - [结果](#结果)
  - [总结](#总结)

## 总览

Morphological Antialiasing(MLAA), 中文一般翻译为形态抗锯齿(反走样), 是一种常见的抗锯齿算法. 其于2009年由Intel的Alexander Reshetov提出, 启发了后续一批基于图像自身形态进行抗锯齿操作的算法例如FXAA和CMAA. 相比传统的基于超采样的抗锯齿算法, MLAA是一种纯粹的后处理算法, 无须法线和深度等信息就可以直接对渲染器的帧缓冲进行抗锯齿处理, 因此这类方法由于即插即得的易用性而得到广泛的应用.

MLAA的思路基于人眼感知的一大特征: 对形状失真的敏感性远强于对颜色失真的敏感性. 因此类似MSAA的想法, MLAA通过一定的策略插值将失真强烈的几何边缘进行模糊, 又保留平滑部分不进行处理, 一方面防止了纹理部分出现额外的失真, 另一方面大大减少了计算量.

而形态抗锯齿的核心是"形态"部分, MLAA先在图片中找到代表几何边缘的部分, 然后将这些边缘分为多种不同的形态模式(pattern), 根据模式实施不同的模糊策略, 这个过程本质上是对边缘矢量的重建和再光栅化的过程. 经过MLAA处理的图片如下图边缘较为平滑, 而内部纹理保持原样, 有效减少了图片失真又不至于产生过多的模糊.

![picture 1](Media/b6d824cef71810b3ee99ee7bc1ec6ac99a9bd83279a194de398c2371b689987e.png)  

### 参考资料

这里参考的核心文章是Reshetov的原始论文"Morphological Antialiasing"和Jimenez一年后发表的"Practical Morphological Anti-Aliasing". 两者的区别在于Reshetov的MLAA是在CPU上实现的, 目的是优化光线追踪渲染的图像, 计算量比较大, 而Jimenez针对光栅化渲染, 以牺牲一部分效果为代价在GPU上以极低的计算量实现了MLAA, 将MLAA的实用性提升了一大截.

这里我的Python实现综合了上面两篇文章. 主体仍然是Reshetov的实现方式, 但使用Jimenez的实现中利用图像来储存临时数据的思路辅助. 此文章的代码仓库的路径如下. 文章为了简洁采用的是提炼的部分代码作为伪代码辅助介绍:

> https://github.com/ZFhuang/MLAA-python

下面是一些可供查阅的辅助资料:

> Intel的MLAA主页
> 
> https://software.intel.com/content/www/cn/zh/develop/articles/morphological-antialiasing-mlaa-sample.html
> 
> 2009到2017形态抗锯齿系列算法的发展
> 
> http://www.iryoku.com/research-impact-retrospective-mlaa-from-2009-to-2017
> 
> Jimenez实现的MLAA的项目主页
> 
> http://www.iryoku.com/mlaa/
> 
> 从零开始的游戏引擎编写之路：形态学抗锯齿
> 
> https://www.bilibili.com/read/cv2269091
> 
> 图形学基础 - 着色 - 空间抗锯齿技术
> 
> https://zhuanlan.zhihu.com/p/363624370

### 流程概览

MLAA分为下面四个大步骤:

1. 查找图片中明显的像素不连续区域作为需要处理的边缘
2. 将这些边缘分类为不同的模式(pattern)
3. 按照不同的模式计算用于融合像素颜色的权重
4. 将像素与周围像素进行按照权重进行混合得到平滑后的结果

## 流程

### 查找边缘

抗锯齿技术处理的目标是图像中边缘部分的锯齿状走样. MLAA首先需要查找出图像中的边缘信息. 在MLAA中, 图像边缘信息的查找相对单个通道进行的, 因此对于彩色图像来说, 需要通过某个方法将其转为单通道形式. 常用的方法是逐通道计算和转为灰度图再计算. 由于常见的图像三个通道的信息可能有很大差异, 因此将彩色图像转为灰度图像后再进行边缘查找是比较合适的算法. Jimenez实现的论文中给出了计算公式:

![picture 1](Media/f707a20e14141c6eb0504a5d70281e8d7c0f52f7d4bb8d3981e140864549286b.png)  

下面是与Jimenez论文中的测试样例相同的太空侵略者的经典敌人, 一张外星人单通道点阵图. 本文后续的流程展示皆以此图为例. 由于实现稍有不同所以后面展示的中间结果会有些许差别, 但最终结果是一样的.

![picture 3](Media/07e1978b59d74dcaf8b611ef6db491f270efe0e38026df0e6507ef4772cfc311.png)  

得到单通道图像后就是查找边缘的步骤了. MLAA将图像的边缘划分为两种: 横向边缘和纵向边缘. 遍历图像中每个像素, 将当前像素与左边和上边相邻的像素做差对比, 当差别大于某个阈值th时认为此像素覆盖边缘. Jimenez的论文中提到对于颜色域是[0,1]的图像来说, th=0.1是比较实用的选择.

对于查找边缘阶段, 可以用一个初始全为0的三通道图片保存边缘信息. 当出现差别的像素处于当前像素左侧时, 我们认为边缘在两个像素相邻的那条边也就是左侧边, 将图片的R通道设置为1; 当出现差别的像素处于当前像素上方时, 边缘处于当前像素上侧, 将图片的G通道设置为1. 一个像素可能同时存在两个边缘, 完成边缘查找阶段后得到的边缘信息图会是由如下红绿黄三色构成的:

![picture 1](Media/248f0c85fafc9485a95a75308c5e7bad442fe48a78dcae06b66a08dd43e52960.png)  

代码如下:

```python
def _find_edges(img, th=0.1):
    buffer = np.zeros((img.shape[0], img.shape[1], 3))
    for y in range(1, img.shape[0]):
        for x in range(0, img.shape[1]):
            if abs(img[y, x]-img[y-1, x]) > th:
                buffer[y, x, 1] = 1
    for y in range(0, img.shape[0]):
        for x in range(1, img.shape[1]):
            if abs(img[y, x]-img[y, x-1]) > th:
                buffer[y, x, 0] = 1
    return buffer
```

### 模式分类



### 权重计算

![picture 2](Media/9161948acff986ed35de5f83f2bde9789b315dfd4cab197b5d1371d667aef5cc.png)  


### 颜色混合

![picture 5](Media/188719e95455a513f03fb2848257a3ef3014db7aac941ffad7629391834d2ae0.png)  

## 结果

## 总结