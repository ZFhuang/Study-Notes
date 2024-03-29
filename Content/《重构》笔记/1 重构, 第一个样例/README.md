# 1 重构, 第一个样例

- [1 重构, 第一个样例](#1-重构-第一个样例)
  - [1.1 起点](#11-起点)
  - [1.2 重构的第一步](#12-重构的第一步)
  - [1.3 分解并重组statement()](#13-分解并重组statement)
  - [1.4 运用多态取代与价格相关的条件逻辑](#14-运用多态取代与价格相关的条件逻辑)


## 1.1 起点

1. 重构是在不改变代码外在行为的前提下对代码进行修改以改进程序的内部结构
1. 太长的函数做了太多的事情是很不好的
2. 差劲的系统很难修改因为很难找到好的修改点, 这导致程序员很容易犯错
3. 通过复制粘贴来完成多个功能很危险, 如果程序要保存很长时间且需要修改, 复制粘贴会造成潜在的威胁, 后来对需求的改变需要同步到多个复制的函数中, 很容易出bug
4. 如果发现需要对程序加一个特性而代码结构很难方便地进行修改, 那么先重构那个程序然后再添加特性

## 1.2 重构的第一步

1. 重构的第一步是为代码建立一组可靠的测试环境
2. 测试需要自动对各种样例进行检验并给出结果, 结果需要自动化指出错误行号等信息减少人工比对
3. 好的测试是重构的根本, 建立一个优良的测试机制会给你的重构必要的保障

## 1.3 分解并重组statement()

1. 代码块越小, 代码的功能就越容易管理, 代码的处理和移动也就越轻松
2. 任何不会被修改的变量都可以作为参数传入新函数
3. 如果只有一个变量会被修改, 那么可以作为函数的返回值
1. 重构时需要不断进行单元测试, 每次进行小修改就进行一下简单的调试, 是的哪怕犯下错误也能很快发现
2. 写出人类容易理解的代码才是优秀的程序员
3. 当不想修改接口时, 先写一个新函数, 然后将其用旧函数来包装可以避免一些接口修改的问题
4. 临时变量只属于函数本身, 容易助长冗长的函数, 且我们容易忘记临时变量的跟踪导致一些意外的错误计算
5. 用函数调用消除临时变量可以让程序更直观, 但是可能会影响性能
6. 在单元测试的时候也应进行时间计算, 最好是分模块的, 这样能够方便进行性能评估. 但是性能优化不要在重构的时候进行, 等到专门的优化时段再来利用留下的信息进行优化

## 1.4 运用多态取代与价格相关的条件逻辑

1. 类的数据应该只属于内部, 最, 对类的数据操作应该在类内进行而不是类外
2. 任何时候都应通过类提供的取值和设值函数来调整类内的变量
3. 每次修改功能后测试的时候应该加多一些暂时的输出确保部件运行正常
4. 重构就应该是反复的”修改-测试“循环, 一步步缓慢朝前推进