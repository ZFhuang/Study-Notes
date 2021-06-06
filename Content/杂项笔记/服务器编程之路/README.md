# 服务器编程之路

- [服务器编程之路](#服务器编程之路)
  - [VSCode与服务器](#vscode与服务器)
    - [最初的连接](#最初的连接)
    - [设置ssh密钥](#设置ssh密钥)
    - [在VSCode上离线安装插件](#在vscode上离线安装插件)
    - [在服务器上进行调试](#在服务器上进行调试)
    - [用Git与服务器协作](#用git与服务器协作)
    - [VSCode实用操作](#vscode实用操作)
    - [Debugger配置](#debugger配置)
    - [命令行调用Debugger](#命令行调用debugger)
  - [设备使用情况](#设备使用情况)
    - [CPU使用情况](#cpu使用情况)
    - [Memory使用情况](#memory使用情况)
    - [GPU使用情况](#gpu使用情况)
  - [Linux常用命令](#linux常用命令)
    - [周期性查看某命令](#周期性查看某命令)
    - [获取文件数量](#获取文件数量)
    - [捕获进程输出](#捕获进程输出)
    - [杀死进程](#杀死进程)
  - [环境配置](#环境配置)
    - [conda环境操作](#conda环境操作)
    - [切换cuda版本](#切换cuda版本)
  - [Tensorflow](#tensorflow)
    - [版本问题](#版本问题)
  - [Pytorch](#pytorch)

记录在无root权限且无法连接外网的服务器上使用ssh连接进行linux编程和深度学习踩过的坑

## VSCode与服务器

### 最初的连接

### 设置ssh密钥

### 在VSCode上离线安装插件

### 在服务器上进行调试

### 用Git与服务器协作

### VSCode实用操作

### Debugger配置

```json
{
    // 典型的debug配置文件如下
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Current Config", // 此debug配置的命名
            "type": "python",
            "request": "launch",
            "program": "./file.py", // 目标文件路径
            "console": "integratedTerminal",
            "args" : ["--port", "1593"] // 需要附着的命令行参数
        },
        {
            "name": "Another config",... // 和上一个大括号同款就可设置其它配置
        }
    ]
}
```

### 命令行调用Debugger

```shell
python -m debugpy
    --listen | --connect
    [<host>:]<port>
    [--wait-for-client]
    [--configure-<name> <value>]...
    [--log-to <path>] [--log-to-stderr]
    <filename> | -m <module> | -c <code> | --pid <pid>
    [<arg>]...
```

## 设备使用情况

### CPU使用情况

```shell
# 直接调用就会自动刷新
$ top # 查看所有进程对cpu的使用情况
$ top -u {name} # 查看用户{name}的进程对cpu的使用情况
$ top -p {pid} # 查看编号为{pid}的进程对cpu的使用情况
```

### Memory使用情况

```shell
$ free  # 以K为单位显示当前内存情况
$ free -m # 以M为单位显示当前内存情况
$ free -h # 以G为单位显示当前内存情况
```

### GPU使用情况

```shell
# 自带的nvidia-smi
$ nvidia-smi
```

```shell
# 有颜色, 需要安装: pip install gpustat
$ gpustat -i  #, 直接调用就会刷新
```

## Linux常用命令

### 周期性查看某命令

```shell
$ watch -n {time} {code}  # 每{time}秒刷新一次{code}信息
```

### 获取文件数量

```shell
$ ls -l | grep "^-" | wc -l # 统计当前目录下文件的个数(不包括目录)
$ ls -lR| grep "^-" | wc -l # 统计当前目录下文件的个数(包括子目录)
$ ls -lR | grep "^d" | wc -l  # 查看某目录下文件夹(目录)的个数(包括子目录)
```

### 捕获进程输出

```shell
$ strace -p {pid} -ewrite | grep 'write(1,' # 当进程在后台运行时使用, 捕获编号为{pid}的进程的控制台输出
```

### 杀死进程

```shell
$ kill {pid}  # 杀死编号为{pid}的进程
$ killall -u {name} # 杀死名为{name}的用户的所有进程  
```

## 环境配置

### conda环境操作

```shell
$ conda list  # 查看当前虚拟环境所装的包
$ conda info -e # 查看当前拥有的环境
$ conda create -n {name} python={version} # 创建名为{name}的带有{version}版本的虚拟环境
$ conda activate {name} # 切换到名为{name}的环境
$ conda create -n {new_name} --clone {old_name} # 复制名为{old_name}的环境
$ conda remove -n {name} --all  # 删除名为{name}的环境
```

### 切换cuda版本

> https://zhuanlan.zhihu.com/p/59123983

```shell
$ cat /usr/local/cuda/version.txt # 查看当前cuda版本
$ cd /usr/local # 查看已安装的cuda
$ vim ~/.bashrc # 在里面修改当前环境变量的路径即可
```

## Tensorflow

### 版本问题

```py
# 遇到2.X版本想要运行1.X版本代码时, 将导入tf的地方替换为下面的代码
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

## Pytorch