- [绪](#绪)
- [cppreference的相关介绍](#cppreference的相关介绍)
- [智能指针总览](#智能指针总览)
- [智能指针与垃圾回收的区别](#智能指针与垃圾回收的区别)
- [unique_ptr的简单实现](#unique_ptr的简单实现)
- [智能指针基类](#智能指针基类)
- [shared_ptr的简单实现](#shared_ptr的简单实现)
- [weak_ptr的简单实现](#weak_ptr的简单实现)
- [make_xxx的简单实现](#make_xxx的简单实现)

## 绪

本篇是尝试对C++11的三种智能指针(unique_ptr, shared_ptr, weak_ptr)进行的复现结果, 智能指针的复现在面试中经常考到, 需要好好熟悉.

才疏学浅, 错漏在所难免, 尤其是智能指针想要全面实现的话属实困难, 各种bug也难以排查, 后续若有新的修改会总结为一篇更好的文章, 望多多包容. 全文 k字, 参考文献和相关资料在每一段的开头给出, 慢慢来吧.

## cppreference的相关介绍

> std::unique_ptr https://en.cppreference.com/w/cpp/memory/unique_ptr
>
> std::shared_ptr https://en.cppreference.com/w/cpp/memory/shared_ptr
>
> std::weak_ptr https://en.cppreference.com/w/cpp/memory/weak_ptr

## 智能指针总览

> 一些基础知识: CppCon 2019: Arthur O'Dwyer "Back to Basics: Smart Pointers" https://youtu.be/xGDLkt-jBJ4
> 
> 智能指针复现的较好样例 https://github.com/SRombauts/shared_ptr
> 
> 一个智能指针的code review, 学习如何找自己的错误: https://codereview.stackexchange.com/questions/163854/my-implementation-for-stdunique-ptr

1. `std::shared_ptr`是核心, 记录了有多少个`std::shared_ptr`指向同一个对象, 计数为0时自动delete. `std::make_shared`用来消除显式使用的new操作, `.get()`用来取得原始指针, `.reset()`用来消掉一个当前的引用计数, `.use_count()`得到目标对象的引用计数. `std::shared_ptr`有两个风险: 引起悬空引用(指针的指针, 用make_shared()优化), 引起循环引用(用`std::weak_ptr`优化)
2. `std::unique_ptr`也是核心, 是对原生指针的独占包装(没有计数器, 几乎零开销), 也有`std::make_unique`(C++14引入)可以生成. `std::unique_ptr`禁止拷贝, 但允许移动到另一个`std::unique_ptr`中
3. `std::weak_ptr`为了解决循环引用的问题而提出, 类似`std::shared_ptr`但是不会增加引用计数. `std::weak_ptr`没有`*`和`->`因此不能操作资源, `.expired()`可以检查目标资源是否被释放, 若释放则返回true. `.lock()`在资源未释放的时候返回一个新的`std::shared_ptr`, 否则返回nullptr
4. 智能指针的释放依赖于作用域, 所以当需要使用更大的生命周期时还是应该考虑手动管理或者在更大的作用域中使用智能指针

## 智能指针与垃圾回收的区别

> 相关讨论: RAII vs. Garbage Collector https://stackoverflow.com/questions/44325085/raii-vs-garbage-collector
> 
> 相关讨论: Garbage collection vs. shared pointers https://stackoverflow.com/questions/4663385/garbage-collection-vs-shared-pointers
> 
> 相关视频: CppCon 2016: Herb Sutter "Leak-Freedom in C++... By Default." https://www.youtube.com/watch?v=JfmTagWcqoE

1. 智能指针的回收通常指的是RAII(资源获取即初始化). 智能指针所采用的引用计数法属于一种垃圾回收技术
2. 智能指针和完整的垃圾回收器旨在解决不同的问题. 垃圾回收是针对内存无目的分配的方案, 其优化目的是内存, 相比之下智能指针关注于具体的资源, 更适合处理锁, 外部资源, 套接字等需要明确处理的内容
2. 垃圾回收器是运行在程序背后的, 有自己不小的开销. 智能指针的额外开销很小
3. 垃圾回收器由于使用标记和清理手法因此可以很好地处理循环引用的问题, 智能指针如果仅仅使用引用计数的话难以处理. weak_ptr也只是优化了这个问题, 需要程序员进行深入的控制
5. 垃圾回收器不期望程序员手动控制对象的回收, 因此失效对象的回收时间是无法预测的, 程序员只能控制需要使用的生命周期(或者使用with获得一定的手动控制能力), 从而难以预测性能. 智能指针则完全由程序员控制
6. 垃圾回收器对于资源的释放有时候有优化, 例如将大批需要释放的资源集中起来一起释放, 这可以提高回收的效率, 智能指针大多只能逐个释放. 但是垃圾回收器的分析和执行仍然有很大的代价
7. 垃圾回收器有时候会遇到缓存优化的问题, 而且回收的时间可能会造成停顿, 智能指针在这点上表现更好, 可以实时处理
8. 垃圾回收器一般在单独的线程中执行, 而智能指针的回收在当前线程的析构函数中执行, 因此智能指针可能导致当前线程的卡顿

## unique_ptr的简单实现

1. unique_ptr只能移动不能复制, 因此是唯一的所有权. 所有不需要分享的指针都应该用这个来代替
2. unique_ptr还有一个默认模板参数是deleter决定析构时的动作. 默认的default_delete仅仅是调用了delete操作, 可以自定义deleter来决定析构时的操作
3. 我们应该像传递raw指针一样传递智能指针, 不用去考虑引用, 右值引用之类的操作
4. 编写的时候注意尽量不要抛出异常, 用noexcept优化速度

```C++
template<typename T>
class UniquePtr {
  using pointer = T*;

private:
  pointer data;

private:
  // 清内存并置空
  void _destroy(UniquePtr& target) noexcept {
      if (target.data != nullptr) {
          delete target.data;
          target.data = nullptr;
      }
  }

public:
  // 用于外部获取指针原始类型
  typedef T ElementType;

  // 默认构造函数, 赋值为nullptr
  UniquePtr() noexcept :data(nullptr) {}

  // 显式构造函数, 为了防止隐式类型转换
  explicit UniquePtr(const pointer& data) noexcept : data(data) {}

  // 析构函数
  ~UniquePtr() noexcept { _destroy(*this); }

  // 移动构造和移动赋值都存在, 用swap实现, 移动后记得清空对方
  UniquePtr(UniquePtr&& moving) noexcept : data(nullptr) {
      swap(moving);
      _destroy(moving);
  }
  // 因为有了nullptr_t构造, 因此可以进行nullptr赋值
  UniquePtr& operator=(UniquePtr&& moving) noexcept {
      if (this != &moving){
          swap(moving);
          _destroy(moving);
      }
      return *this;
  };

  // 拷贝构造和拷贝赋值都被禁止, 采用const&来写就是万能引用
  UniquePtr(const UniquePtr&) noexcept = delete;
  UniquePtr& operator=(const UniquePtr&) noexcept = delete;
  // 仅允许使用nullptr进行拷贝赋值, 因为这相当于reset
  UniquePtr& operator=(std::nullptr_t) noexcept {
      reset();
      return *this;
  }

  // 显式bool转换函数
  explicit operator bool() const noexcept { return data != nullptr; }

  // 只能指针都需要模仿原生指针, 注意不要对nullptr操作
  T& operator*() const noexcept { assert(data != nullptr); return *data; }
  pointer operator->() const noexcept { assert(data != nullptr); return data; }
  pointer get() const noexcept { return data; }

  // 用于交换指针的成员函数, 非常非常常用
  void swap(UniquePtr& other) noexcept {
      std::swap(data, other.data);
  }
  void swap(UniquePtr&& other) noexcept {
      std::swap(data, other.data);
  }

  // 将内部指针置为外部值并删去当前值, 注意防止自我赋值
  void reset(pointer p = nullptr) noexcept {
      swap(UniquePtr(p));
  }
};
```

## 智能指针基类

1. 由于shared_ptr和weak_ptr都有一个堆储存的计数器来维护计数进行内存回收, 为了编写的方便将其写为一个基类来继承
2. 由于shared_ptr和weak_ptr的计数器是共享的, 有可能被多线程竞争修改, 因此需要有额外的mutex来保护, 所有堆counter的修改都需要经过mutex原子保护

```C++
class PtrBase {
public:
  // stl实现的智能指针还会在Counter中放入注册好的deleter
  struct Counter {
      int uses = 0;
      int weaks = 0;
  };
  using p_counter = Counter*;
  using p_mutex = std::mutex*;
  // 堆计数器用来在多个ptr间共享
  p_counter counter;
  // 堆内存mutex保证线程安全, 计数器为nullptr时才回收
  p_mutex mutex;

protected:
  // 这里用到了委托构造的技巧. 需要new计数器和互斥量
  // 注意由于用到new所以可能产生异常. 让异常逃离构造函数很麻烦, 因此用nothrow然后自己处理
  PtrBase() noexcept : PtrBase(
      new (std::nothrow) Counter(),
      new (std::nothrow) std::mutex())
  {}
  PtrBase(std::nullptr_t) noexcept : PtrBase() {}
  PtrBase(p_counter counter, p_mutex mutex) noexcept :
      counter(counter),
      mutex(mutex)
  {}

  void increase_shared_count() noexcept {
      if (counter != nullptr) {
          mutex->lock();
          ++(counter->uses);
          mutex->unlock();
      }
  }

  void increase_weak_count() noexcept {
      if (counter != nullptr) {
          mutex->lock();
          ++(counter->weaks);
          mutex->unlock();
      }
  }

  // 只要share计数为0就返回给指针本身以回收目标对象的内存
  bool reduce_shared_count() noexcept {
      bool is_zero = true;
      if (counter != nullptr) {
          mutex->lock();
          --(counter->uses);
          if (counter->uses > 0) {
              is_zero = false;
          }
          mutex->unlock();
      }
      return is_zero;
  }

  // 只有当两种引用都为0时才可以回收计数器本身的内存, 记得所有对堆内存的修改都加锁
  void reduce_weak_count() noexcept {
      if (counter != nullptr) {
          mutex->lock();
          if (counter->weaks > 0) {
              --(counter->weaks);
          }
          if (counter->uses == 0 && counter->weaks == 0) {
              delete counter;
              counter = nullptr;
          }
          mutex->unlock();
      }
  }

  void check_mutex() noexcept {
      if (counter == nullptr) {
          delete mutex;
          mutex = nullptr;
      }
  }

  // new失败的时候做的补救措施
  void revert() noexcept {
      if (mutex != nullptr) {
          reduce_shared_count();
          reduce_weak_count();
          delete mutex;
          mutex = nullptr;
      }
      if (counter != nullptr) {
          delete counter;
          counter = nullptr;
      }
  }

  void swap(PtrBase& other) noexcept {
      std::swap(counter, other.counter);
      std::swap(mutex, other.mutex);
  }
};
```

## shared_ptr的简单实现

1. shared_ptr需要一个间接层处理引用计数的问题, 因此带来了额外的开销, unique_ptr则完全没有额外的空间开销
2. 对于性能不敏感的情况, 最好不要使用原始指针
3. 建议不要对某个对象进行两次以上的shared, 我们的脑子处理不了太多的共享, 用weak代替
4. stl中通过让自己的类继承enable_shared_from_this类, 我们可以生成指向自身this的shared_ptr
5. 这个问题是由于非侵入式访问的标准库设计哲学, shared_ptr的计数器和对象本身是分离的, 如果在类中对this构造一个shared_ptr, 那么产生的是第二个计数器, 和初始化两次shared_ptr的效果是一样的, 并不是拷贝. 因此在类中这个构造函数结束后, 这个对象(自己)就会被调用析构, 然后一切都boom了
6. enable_shared_from_this则通过weak_ptr安全地生成了一个自己的shared_ptr, 防止了析构问题. 这种现象常出现在多线程的回调中, 其实不是很常见
7. stl实现的make_shared还支持了优化, 让目标对象和间接层连续储存从而减少了new和delete的开销
8. stl的unique_ptr可以被赋值给shared_ptr
9. shared_ptr并非完美, 例如用同一个原生指针构造两个智能指针的话, 目标内存会被重复析构而报错, 因此最好避免这种直接的指针操作

```C++
template<typename T>
class SharedPtr : public PtrBase {
  using pointer = T*;
  // 需要和WeakPtr形成friend方便两者之间的转型
  friend class WeakPtr<T>;

private:
  // 原生指针
  pointer data;

private:
  // 先减少计数, 如果为0则释放资源
  void _destroy(SharedPtr& target) noexcept {
      if (data != nullptr) {
          if (target.reduce_shared_count()) {
              delete target.data;
              target.data = nullptr;
          }
          target.check_mutex();
      }
  }

  // 给WeakPtr用的构造
  SharedPtr(const WeakPtr<T>& wptr) noexcept : data(wptr.data), PtrBase(wptr.counter, wptr.mutex) {
      increase_shared_count();
  }

public:
  // 用于外部获取指针原始类型
  typedef T ElementType;
  
  // 默认构造函数, 全部赋为nullptr
  SharedPtr() noexcept : data(nullptr), PtrBase() {};

  // 记得显式以防止隐式类型转换
  explicit SharedPtr(const pointer& data) noexcept :
      data(data), PtrBase() {
      // nullptr代表空间申请失败
      if (counter == nullptr || mutex == nullptr) {
          this->data = nullptr;
          revert();
      }
      if (data != nullptr) {
          increase_shared_count();
      }
  }

  ~SharedPtr() noexcept {
      _destroy(*this);
  }

  // 拷贝构造, shared_ptr拷贝后会将计数器+1
  SharedPtr(const SharedPtr& copy) noexcept : data(copy.data), PtrBase(copy.counter, copy.mutex) {
      if (data != nullptr) {
          increase_shared_count();
      }
  }
  // 拷贝赋值, 采用copy-swap写法, 由于右值引用的存在, 折叠式写法会造成二义性
  // 旧的内存会由于tmp的析构而释放, 新的内存的申请也在tmp的拷贝构造中完成了
  SharedPtr& operator=(const SharedPtr& copy) noexcept {
      SharedPtr tmp(copy);
      swap(tmp);
      return *this;
  }
  // 用nullptr赋值时相当于清空
  SharedPtr& operator=(std::nullptr_t) noexcept {
      _destroy(*this);
      return *this;
  }

  // 移动构造函数, 由于是构造所以可以直接夺取指针内容
  // 析构的时候由于目标是nullptr所以自然结束掉
  SharedPtr(SharedPtr&& moving) noexcept : data(nullptr), PtrBase() {
      swap(moving);
      _destroy(moving);
  }
  // 移动赋值函数
  SharedPtr& operator=(SharedPtr&& moving) noexcept {
      if (this != &moving) {
          swap(moving);
          _destroy(moving);
      }
      return *this;
  }

  // 老三样
  pointer operator->() const noexcept { assert(data != nullptr); return data; }
  T& operator*() const noexcept { assert(data != nullptr); return *data; }
  pointer get() const noexcept { return data; }

  // 用于交换指针的成员函数
  void swap(SharedPtr& other) noexcept {
      std::swap(data, other.data);
      PtrBase::swap(other);
  }

  void swap(SharedPtr&& other) noexcept {
      std::swap(data, other.data);
      PtrBase::swap(other);
  }

  // 通过与新构造的对象交换来简化代码
  void reset(pointer p = nullptr) noexcept {
      swap(SharedPtr(p));
  }

  // 返回当前计数器次数
  int use_count() const noexcept { assert(counter != nullptr); return counter->uses; }

  explicit operator bool() const noexcept { return data != nullptr; }
};
```

## weak_ptr的简单实现

1. weak_ptr的实现与shared_ptr类似, 只是修改的是weak计数
2. 不允许直接从原始指针构造, 必须绑定在shared_ptr上
3. 当share计数为0时, weak_ptr失效
4. weak_ptr不能用来直接操作目标, 只有当指针有效的时候, 通过lock()函数构造一个shared_ptr才能进行操作, 无效的时候lock返回nullptr

```C++
class WeakPtr : public PtrBase {
  using pointer = T*;
  friend class SharedPtr<T>;

private:
  pointer data;

private:
  void _destroy(WeakPtr& target) noexcept {
      if (data != nullptr) {
          target.reduce_weak_count();
          target.check_mutex();
      }
  }

public:
  // 用于外部获取指针原始类型
  typedef T ElementType;
  WeakPtr() noexcept : data(nullptr), PtrBase() {}
  ~WeakPtr() noexcept { _destroy(*this); }
  WeakPtr(const SharedPtr<T>& sptr) noexcept :data(sptr.data), PtrBase(sptr.counter, sptr.mutex) {
      if (data != nullptr) {
          increase_weak_count();
      }
  }
  WeakPtr& operator=(const SharedPtr<T>& copy) noexcept {
      WeakPtr<T> tmp(copy);
      swap(tmp);
      return *this;
  }
  WeakPtr(const WeakPtr& copy) noexcept :data(copy.data), PtrBase(copy.counter, copy.mutex) {
      if (data != nullptr) {
          increase_weak_count();
      }
  }
  WeakPtr& operator=(const WeakPtr& copy) noexcept {
      WeakPtr tmp(copy);
      swap(tmp);
      if (data != nullptr) {
          increase_weak_count();
      }
      return *this;
  }
  WeakPtr& operator=(std::nullptr_t) noexcept {
      reset();
      return *this;
  }

  WeakPtr(WeakPtr&& moving) noexcept : data(nullptr), PtrBase() {
      swap(moving);
      _destroy(moving);
  }
  WeakPtr& operator=(WeakPtr&& moving) noexcept {
      if (this != &moving) {
          swap(moving);
          _destroy(moving);
      }
      return *this;
  }

  SharedPtr<T> lock() noexcept {
      if (expired()) {
          return SharedPtr<T>(nullptr);;
      }
      else {
          return SharedPtr<T>(*this);
      }
  }

  void reset() noexcept {
      swap(WeakPtr());
  }

  void swap(WeakPtr& other) noexcept {
      std::swap(data, other.data);
      PtrBase::swap(other);
  }
  
  void swap(WeakPtr&& other) noexcept {
      std::swap(data, other.data);
      PtrBase::swap(other);
  }

  int use_count() const noexcept { assert(counter != nullptr);  return counter->uses; }
  bool expired() const noexcept { return counter->uses == 0; }
};
```

## make_xxx的简单实现

主要就是使用完美转发和变长参数来无损包装new操作, 从而让new不用暴露在用户面前

```C++
template<typename T, typename... Args>
inline UniquePtr<T> MakeUnique(Args&&... args) {
  return UniquePtr<T>(new T(std::forward<Args>(args)...));
}

template<typename T, typename... Args>
inline SharedPtr<T> MakeShared(Args&&... args) {
  return SharedPtr<T>(new T(std::forward<Args>(args)...));
}
```