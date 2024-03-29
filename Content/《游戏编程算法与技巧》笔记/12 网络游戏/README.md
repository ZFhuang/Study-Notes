# 12 网络游戏

## 各种协议

1. IP: 传输层协议. 将数据从一个IP地址传到另一个IP地址. 分为IPv4(32位地址)和IPv6(128位地址)两种
2. ICMP: 网络层协议. 不用于数据传输, 而主要用于发送回声包来测量两台机器间的延迟. 发送者将当前的时间戳放入数据帧, 然后接收者原样发回, 通过接收到的时间和之前放入的时间戳对比计算延迟时间. ICMP通过校验和来确保可靠
3. TCP: 网络层协议. 基于连接的, 可靠的, 保证顺序的协议. 两台计算机需要先握手建立连接, 然后传输中通过不断校验与重发保证数据包的可靠和有序. 对于高实时性要求的游戏来说, TCP的延迟太大了
4. UDP: 网络层协议. 无连接, 不可靠, 因此会出现丢包问题导致游戏体验变差. 而且为了在不可靠传输中保证一定的顺序性, 通常在UDP数据包中增加一些额外数据, 例如顺序号, 让接收者可以对顺序可靠性增加一定控制. UDP是大多数游戏所采用的网络层协议
5. 处于折中, 也可以对游戏关键数据用TCP传输, 其余数据用UDP传输

## 服务器/客户端模型

1. 由一台服务器和多台客户端组成, 也称为中心型结构
2. 服务器认为是游戏的权威判断者, 客户端的所有关键行为都需要发送给服务器, 由服务器计算, 验证行为是否合法并计算行为造成的后果, 然后通知给相关的其它客户端
3. 因此游戏的很多逻辑判断实际上处于服务器上, 需要实现单人模式的游戏应该设计将单人模式作为此模型中一种特殊的多人模式(服务器和客户端处于同台机器上)开发
4. 为了优化此模型的延迟, 一般会在客户端根据最近服务器返回的数据进行插值, 从而减少服务器延迟对用户体验的影响. 也就是在本地同样进行一部分服务器上的判断用于流畅地渲染, 一旦服务器返回的结果与当前客户端上模拟的结果冲突, 则将客户端的结果矫正
5. 当服务器和客户端在同台机器上时, 作为服务器的机器上的玩家会有主机优势, 因为延迟为0
6. 如果服务器崩溃那么所有客户端都会崩溃
7. 一般的解决方法是设立独立的专用服务器, 既平衡了延迟问题, 又减少了崩溃的可能性

## 点对点模型

1. 所有客户端都连接到其它客户端, 也称为帧同步模型, 常见于RTS等低实时要求的游戏
2. 点对点模型将网络更新划分为150ms-200ms的回合(帧)更新, 每个玩家的操作都保存在队列里, 等到回合结束时一起执行, 因此很多RTS网战会有操作延迟
3. 点对点模型的缺点是只要一个玩家计算出现延迟, 所有玩家都需要等待那个玩家的帧到达
4. 点对点模型的好处是需要传输的数据较少, 只有玩家自己的操作而已
5. 点对点模型需要保证每个玩家在获得相同的帧信息后, 都会模拟出完全相同的结果, 因此基于随机性的游戏逻辑比较闹做

## 作弊

1. 信息作弊: 玩家通过读取内存得到本不应得到的信息. 对于服务器/客户端模型, 可以通过限制服务器下方的信息数量来减少作弊可能性. 点对点模型无法进行相应限制, 一种方法是额外安装反作弊程序监视其它读取内存的进程
2. 游戏状态作弊: 常见于玩家作为服务器的时候, 直接修改游戏数据来胜利. 对抗方法出了反作弊程序外, 还应该对客户端对服务器发送的指令进行检查
3. 中间人攻击: 通过拦截客户端与服务器间传输的信息并修改, 大多数上述反作弊方法都无效, 一种有效解决方法是对传输的数据包进行加密防止篡改. 但加密算法对大多数游戏来说都过重, 一般只保护游戏登入登出的部分