# 并行作业选题 - PCFG口令猜测算法  
## 该分支利用SIMD对口令猜测算法的MD5哈希部分做了优化，修改的文件有：
- `main.cpp`
- `md5.cpp`
- `md5.h`
- `correctness.cpp`
## 也有新增的文件，这些文件的代码实现了SIMD 8路并行化
- `md5_8.cpp`
- `md5_8.h`
- `correctness_8.cpp`
