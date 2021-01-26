# P2 数据拟合

![picture 4](Media/be4219c3a272dd8179c0f851af160d5a2afedc490b1980701dc41c58c22dc5d3.png)  

## 病态问题

病态系统中, 系统元素的一点点扰动就会导致系统求解出来的结果非常不同, 这本质上是因为在此问题附近解的变化率(导数)非常大

![picture 5](Media/af380092738a1d5a3a407cf732d2f1a571af90bdc4be92cb315b4dde0c2c163a.png)  

病态问题在数学上用矩阵的条件数来刻画, 条件数大, 也就是特征值相差很大, 那么系统就不稳定

![picture 6](Media/04b1051cdb892f6829804c591abf2e7b7ceedf26eb88ad965569189864d7c664.png)  

多项式插值和幂函数插值是病态的

![picture 1](Media/2f6967d28bc9a12fbd106e9d851a4e3c634c210c4477553c40ab065f6692d00b.png)  

