# Chapter16 Computer Animation 计算机动画

- [Chapter16 Computer Animation 计算机动画](#chapter16-computer-animation-计算机动画)
  - [16.1 Principles of Animation 动画的原则](#161-principles-of-animation-动画的原则)
    - [16.1.1 Timing 时间控制](#1611-timing-时间控制)
    - [16.1.2 Action Layout 动作布局](#1612-action-layout-动作布局)
    - [16.1.3 Animation Techniques 动画技术](#1613-animation-techniques-动画技术)
    - [16.1.4 Animator Control vs. Automatic Methods 动画师控制对比自动化方法](#1614-animator-control-vs-automatic-methods-动画师控制对比自动化方法)
  - [16.2 Keyframing 关键帧](#162-keyframing-关键帧)
    - [16.2.1 Motion Controls 动作控制](#1621-motion-controls-动作控制)

这一章介绍了计算机动画相关的内容, 主要介绍了动画的基本概念, 动画之间的插值, 几何变形, 角色层级动画, 基于物理的动画, 生成式动画和对象组动画这几个领域. 对于这些领域都只介绍了最基础的内容, 想要了解必须阅读其它, 难度不高, 当作科普看待即可.

## 16.1 Principles of Animation 动画的原则

1930年的时侯迪士尼提出了著名的动画十二原则: 

1. 挤压与伸展（Squash and stretch）
2. 预期动作（Anticipation）
3. 演出方式（Staging）
4. 接续动作与关键动作（Straight ahead action and pose to pose）
5. 跟随动作与重叠动作（Follow through and overlapping action）
6. 渐快与渐慢（Slow in and slow out）
7. 弧形（Arcs）
8. 附属动作（Secondary action）
9. 时间控制（Timing）
10. 夸张（Exaggeration）
11. 立体造型（Solid drawing skill）
12. 吸引力（Appeal）

尽管这十二原则一般是给动画师参考的, 但是其中的很多效果需要由计算机来辅助实现, 因此我们也需要对其有一定的了解.

### 16.1.1 Timing 时间控制

时间控制, 也就是动画的时间节奏. 直觉上来说就是动作的停顿和快慢, 质量大的物体我们希望动作慢, 反之动作快.

### 16.1.2 Action Layout 动作布局

也就是如何安排动作来吸引观众的目光, 一方面是需要对动作设置一些明显的预先动作和结束动作, 另一方面是给动作加上很多次要动画, 例如过渡的重叠帧, 保证不同动画之间的自然过渡.

### 16.1.3 Animation Techniques 动画技术

这主要指我们希望动画中存在一些挤压和拉伸夸张化动作, 常见的是柔软物体加速度急剧改变时发生的明显形变. 另一方面是我们希望动作的变化有缓入缓出, 并且尽量避免出现直线动作因为直线动作并不自然. 这些设计需要动画师有高超的技巧, 并且也需要它们对图形学有些理解才能更好地使用图形学工具进行动画设计.

### 16.1.4 Animator Control vs. Automatic Methods 动画师控制对比自动化方法

动画师能够设计出精细生动的动画, 自动化方法难以达到动画师的效果. 因此更重要的是设计出图形软件方便动画师更好更直观地进行动画设计, 既要利用自动化方法补全一些不重要的部分, 又要给动画师足够的自由设计出生动的动作. 自由的设计自然带来复杂的操作, 这又需要我们这些开发者在这两者之间做出平衡.

## 16.2 Keyframing 关键帧

关键帧是用于指示出

### 16.2.1 Motion Controls 动作控制