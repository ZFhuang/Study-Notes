# OpenMesh教程文档

- [OpenMesh教程文档](#openmesh教程文档)
  - [前言](#前言)
  - [作业框架代码不完全导览](#作业框架代码不完全导览)
    - [GUI文件夹](#gui文件夹)
      - [main.cpp](#maincpp)
      - [surfacemeshprocessing.cpp & h](#surfacemeshprocessingcpp--h)
      - [MainViewerWidget.cpp & h](#mainviewerwidgetcpp--h)
      - [MeshViewerWidget.cpp & h](#meshviewerwidgetcpp--h)
    - [MeshViewer文件夹](#meshviewer文件夹)
      - [MeshDefinition.cpp & h](#meshdefinitioncpp--h)
  - [First Steps - Building a cube 第一步-建一个立方体](#first-steps---building-a-cube-第一步-建一个立方体)
  - [Using iterators and circulators 使用迭代器和环绕器](#using-iterators-and-circulators-使用迭代器和环绕器)
  - [Using (custom) properties 使用自定义的网格属性](#using-custom-properties-使用自定义的网格属性)
  - [Using STL algorithms 使用stl算法](#using-stl-algorithms-使用stl算法)
  - [Using standard properties 使用标准属性](#using-standard-properties-使用标准属性)
  - [实战网格遍历](#实战网格遍历)

## 前言

来自[官方文档实例代码](https://www.graphics.rwth-aachen.de/media/openmesh_static/Documentations/OpenMesh-7.0-Documentation/a03951.html)

代码的可视化使用中科大傅孝明老师的[数字几何处理课程](http://staff.ustc.edu.cn/~fuxm/course/2020_Spring_DGP/index.html)提供的[作业框架](http://staff.ustc.edu.cn/~fuxm/code/index.html#sec_surface_framework)

由于使用框架来进行很多处理, 因此跳过官方文档第一步的安装部分

## 作业框架代码不完全导览

### GUI文件夹

保存与GUI等相关的函数

#### main.cpp
- main(int argc, char* argv[])
  - 主函数, 负责最基础的初始化

#### surfacemeshprocessing.cpp & h

初始化界面的部分, 主要调用MainViewerWidget的函数

- SurfaceMeshProcessing::CreateActions(void)
  - 负责初始化一系列将被加入GUI的元件并为其绑定好相关的接口函数调用和文字图标等
- SurfaceMeshProcessing::CreateMenus(void)
  - 菜单栏的初始化, 负责实际加入各个Actions

#### MainViewerWidget.cpp & h

一系列能从图形界面按钮调用的函数接口

- MainViewerWidget::Open(void)
  - 打开文件按钮的接口, 得到文件名
- MainViewerWidget::OpenMeshGUI(const QString& fname)
  - 将网格文件读取的进一步接口, 转接渲染接口类的MeshViewerWidget

#### MeshViewerWidget.cpp & h

一系列与渲染界面有关的进一步的接口

- MeshViewerWidget::LoadMesh(const std::string & filename)
  - 调用真正的网格读取的接口MeshTools::ReadMesh, 读取成功的时候刷新渲染

### MeshViewer文件夹

保存与算法真正有关的函数

#### MeshDefinition.cpp & h

关于网格本身的类, 对Mesh类进行了方法封装

- MeshTools::ReadMesh(Mesh& mesh, const std::string& filename)
  - 从OpenMesh能直接处理的文件中直接读取几何和拓扑到Mesh中
- MeshTools::ReadOBJ(Mesh& mesh, const std::string& filename)
  - 读取OBJ文件的函数, 通过对文件进行两次扫描来读取顶点和面, 然后写入Mesh中

## First Steps - Building a cube 第一步-建一个立方体

OpenMesh的三角网格类需要通过OpenMesh::TriMesh_ArrayKernelT<MeshTraits>初始化, 也就是网格需要选择好内核和模板类, 下面设这个类的实例叫mesh

通过VHandle得到网格本身的顶点数组, 这是为了加速后面的设置网格面的操作

mesh.add_vertex(Mesh::Point(x,y,z))用来往网格中添加顶点同时返回handle以供后面操作

常用std::vector<Mesh::VertexHandle> face_vhandle来设置网格的面片

将所需的连接关系的Vhandle按照顺序写入push_back到face_vhandle作为设置网格面片的缓冲, 设置好一个面就mesh.add_face(face_vhandles)应用这个面, 随后face_vhandles.clear()清除当前的缓冲

然后调用update显示

## Using iterators and circulators 使用迭代器和环绕器

Mesh::VertexIter是顶点迭代器, 会访问所有顶点

Mesh::VertexVertexIter内层迭代器(环绕器), 此迭代器会按照半边遍历顶点周围邻接的顶点, 详见虎书

重心平滑就是遍历整个网格, 将所有顶点都移动到它环绕顶点生成的坐标上, 重复多次就会平滑


## Using (custom) properties 使用自定义的网格属性

可以通过如OpenMesh::VPropHandleT<Mesh::Point>来新建一个与网格相关的缓冲区, 这一句是与网格顶点相关的缓冲区

然后通过mesh.add_property(tmp)让其与网格绑定

绑定后我们可以用mesh.property(tmp, *v_it) 来取出指针v_it对应的网格顶点对应的属性tmp并直接进行相应的计算

## Using STL algorithms 使用stl算法

模板和STL, 先跳过

## Using standard properties 使用标准属性

可以用request_xxx()来让网格增加某个本来没有但是存在于列表中的标准属性例如法线属性, 然后用update_xxx()来刷新计算这个属性, 用的时候直接xxx()就可以返回其内容, 使用后用release_xxx()来删去这个属性.

## 实战网格遍历