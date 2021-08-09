# 形态抗锯齿MLAA详解与Python实现

- [形态抗锯齿MLAA详解与Python实现](#形态抗锯齿mlaa详解与python实现)
  - [总览](#总览)
    - [参考资料](#参考资料)
    - [流程概览](#流程概览)
  - [流程](#流程)
    - [查找边缘](#查找边缘)
    - [模式分类](#模式分类)
    - [权重计算](#权重计算)
    - [颜色混合](#颜色混合)
  - [结果](#结果)
  - [展望](#展望)

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

得到图片边缘之后, MLAA论文中将边缘视作走样并分为三个模式: L型, Z型, U型. 下面是Reshetov对这三种模式给出的示意图:

![picture 6](Media/f5a5369c537fdc658c71f4f25e64a237e01b87ebe6d23a13f5fbef8fd0ee98a3.png)  

但想要用程序直接寻找着三种模式是比较困难的, 所以这里我们对模式搜索算法进行优化, 将所有模式都转为长边与短边的组合. 注意到这些走样模式都是由长度为1的一到两条短边与长度未知的一条长边组成, 所有的模式都需要长边的存在, 因此我们将长边的出现视作模式搜索的起点, 当模式遇到短边或达到尽头时模式结束, 所以将短边或空像素视作模式搜索的终点, 从而将所有模式转换为两个子模式的组合. 

然后首先将模式搜索分为基于X和基于Y两种搜索顺序, 以X优先搜索为例, 当遍历发现G通道的值为1时, 也就是当前像素上方有横边存在, 认为遇见了走样, 判断上面相邻像素和自身像素的R通道是否有1存在. 若上方相邻像素R通道为1, 此走样的前半段定为B型, 表示长边在短边下方, 若当前像素R通道为1, 此走样前半段定为T型, 表示长边在短边上方, 若当前和上方像素R通道都为1, 定为H型, 表示长边的上下都有短边, 若R通道都为0, 此走样前半段定为L型, 表示形如原论文的L走样, 即一侧缺少短边. 

完成了前半段的搜索后就开始后半段的搜索, 关注点在于计算出走样的长度和后半段的走样模式. 当遍历途中的像素或上方像素的R通道为1时, 表示这段走样来到了终点, 记录下走样所经过的像素数量就是走样的长度, 然后用和起点处相同的判断模式判断出终点处的走样属于TBHL四个模式中的某一个, 记录下来.

熟悉了这个流程后再看下面的两种典型走样情况, 第一个走样是原论文的Z型走样, 经过上面的拆解变为了TB型走样, 第二个走样是原论文的L型走样, 经过拆解变为了LB型走样. 图的下面是对应搜索的代码, 基于X优先遍历搜索完走样模式后, 再以类似的方法按Y搜索一次走样模式, 保存在一个列表里即可.

![picture 8](Media/cc5e4246871a2a28f00edbff156299fe73fe4eacf3d6f7f5b58da484a609ddf1.png)  

```python
def _find_aliasings_x(img_edges):
    list_aliasings = []
    mask = np.zeros((img_edges.shape[0], img_edges.shape[1], 1))
    for y in range(1, img_edges.shape[0]):
        for x in range(0, img_edges.shape[1]):
            if mask[y, x] == 0:
                if img_edges[y, x, 1] == 1:
                    if img_edges[y, x, 0] == 1 and img_edges[y-1, x, 0] == 1:
                        start_pattern = 'H'
                    elif img_edges[y, x, 0] == 1:
                        start_pattern = 'T'
                    elif img_edges[y-1, x, 0] == 1:
                        start_pattern = 'B'
                    else:
                        start_pattern = 'L'
                    dis, end_pattern, mask = _cal_aliasing_info_x(
                            img_edges, x+1, y, mask)
                    list_aliasings.append(
                        [y, x, dis, start_pattern+end_pattern])
    return list_aliasings

def _cal_aliasing_info_x(img_edges, start_x, start_y, mask):
    dis = 1
    for x in range(start_x, img_edges.shape[1]):
        if img_edges[start_y, x, 0] == 1 and img_edges[start_y-1, x, 0] == 1:
            pattern = 'H'
            return dis, pattern, mask
        if img_edges[start_y, x, 0] == 1:
            pattern = 'T'
            return dis, pattern, mask
        if img_edges[start_y-1, x, 0] == 1:
            pattern = 'B'
            return dis, pattern, mask
        if img_edges[start_y, x, 1] == 0:
            break
        mask[start_y, x] = 1
        dis+=1
    pattern = 'L'
    return dis, pattern, mask
```

### 权重计算



![picture 2](Media/9161948acff986ed35de5f83f2bde9789b315dfd4cab197b5d1371d667aef5cc.png)  


### 颜色混合

![picture 5](Media/188719e95455a513f03fb2848257a3ef3014db7aac941ffad7629391834d2ae0.png)  

## 结果

## 展望