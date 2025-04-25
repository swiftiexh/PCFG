#include <iostream>
#include <string>
#include <cstring>
//用NEON实现SIMD并行化
#include <arm_neon.h>
#include <array>


using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

//修改：SIMD一次处理四个32位整数
#define F_NEON(x, y, z) vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32((x)), (z)))
#define G_NEON(x, y, z) vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32((z))))
#define H_NEON(x, y, z) veorq_u32(veorq_u32((x), (y)), (z))
#define I_NEON(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32((z))))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT_NEON(x, n) vsliq_n_u32(vshrq_n_u32((x), 32 - (n)), (x), (n))

#define FF_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(F_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define GG_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(G_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define HH_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(H_NEON(b, c, d), vaddq_u32(x,vdupq_n_u32(ac)))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

#define II_NEON(a, b, c, d, x, s, ac) { \
  a = vaddq_u32(a, vaddq_u32(I_NEON(b, c, d), vaddq_u32(x, vdupq_n_u32(ac)))); \
  a = ROTATELEFT_NEON(a, s); \
  a = vaddq_u32(a, b); \
}

void MD5Hash_SIMD(array<std::string,4> input, bit32 state[4][4]);