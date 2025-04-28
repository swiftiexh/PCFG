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
//要实现8个32位并行，但NEON寄存器只有128位，所以需要两个拼起来
typedef struct {
    uint32x4_t val[2];  // 两个4x32bit，合起来8x32bit
} uint32x8_t;


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

//修改：SIMD一次处理8个32位整数
#define F_NEON(x, y, z) ( (uint32x8_t){ \
    vorrq_u32(vandq_u32((x).val[0], (y).val[0]), vandq_u32(vmvnq_u32((x).val[0]), (z).val[0])), \
    vorrq_u32(vandq_u32((x).val[1], (y).val[1]), vandq_u32(vmvnq_u32((x).val[1]), (z).val[1])) \
} )

#define G_NEON(x, y, z) ( (uint32x8_t){ \
    vorrq_u32(vandq_u32((x).val[0], (z).val[0]), vandq_u32((y).val[0], vmvnq_u32((z).val[0]))), \
    vorrq_u32(vandq_u32((x).val[1], (z).val[1]), vandq_u32((y).val[1], vmvnq_u32((z).val[1]))) \
} )

#define H_NEON(x, y, z) ( (uint32x8_t){ \
    veorq_u32(veorq_u32((x).val[0], (y).val[0]), (z).val[0]), \
    veorq_u32(veorq_u32((x).val[1], (y).val[1]), (z).val[1]) \
} )

#define I_NEON(x, y, z) ( (uint32x8_t){ \
    veorq_u32((y).val[0], vorrq_u32((x).val[0], vmvnq_u32((z).val[0]))), \
    veorq_u32((y).val[1], vorrq_u32((x).val[1], vmvnq_u32((z).val[1]))) \
} )

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
#define ROTATELEFT_NEON(x, n) ( (uint32x8_t){ \
    vsliq_n_u32(vshrq_n_u32((x).val[0], 32 - (n)), (x).val[0], (n)), \
    vsliq_n_u32(vshrq_n_u32((x).val[1], 32 - (n)), (x).val[1], (n)) \
} )


#define FF_NEON(a, b, c, d, x, s, ac) { \
    a.val[0] = vaddq_u32(a.val[0], vaddq_u32(F_NEON(b, c, d).val[0], vaddq_u32((x).val[0], vdupq_n_u32(ac)))); \
    a.val[0] = ROTATELEFT_NEON(a, s).val[0]; \
    a.val[0] = vaddq_u32(a.val[0], b.val[0]); \
    a.val[1] = vaddq_u32(a.val[1], vaddq_u32(F_NEON(b, c, d).val[1], vaddq_u32((x).val[1], vdupq_n_u32(ac)))); \
    a.val[1] = ROTATELEFT_NEON(a, s).val[1]; \
    a.val[1] = vaddq_u32(a.val[1], b.val[1]); \
}

#define GG_NEON(a, b, c, d, x, s, ac) { \
    a.val[0] = vaddq_u32(a.val[0], vaddq_u32(G_NEON(b, c, d).val[0], vaddq_u32((x).val[0], vdupq_n_u32(ac)))); \
    a.val[0] = ROTATELEFT_NEON(a, s).val[0]; \
    a.val[0] = vaddq_u32(a.val[0], b.val[0]); \
    a.val[1] = vaddq_u32(a.val[1], vaddq_u32(G_NEON(b, c, d).val[1], vaddq_u32((x).val[1], vdupq_n_u32(ac)))); \
    a.val[1] = ROTATELEFT_NEON(a, s).val[1]; \
    a.val[1] = vaddq_u32(a.val[1], b.val[1]); \
}

#define HH_NEON(a, b, c, d, x, s, ac) { \
    a.val[0] = vaddq_u32(a.val[0], vaddq_u32(H_NEON(b, c, d).val[0], vaddq_u32((x).val[0], vdupq_n_u32(ac)))); \
    a.val[0] = ROTATELEFT_NEON(a, s).val[0]; \
    a.val[0] = vaddq_u32(a.val[0], b.val[0]); \
    a.val[1] = vaddq_u32(a.val[1], vaddq_u32(H_NEON(b, c, d).val[1], vaddq_u32((x).val[1], vdupq_n_u32(ac)))); \
    a.val[1] = ROTATELEFT_NEON(a, s).val[1]; \
    a.val[1] = vaddq_u32(a.val[1], b.val[1]); \
}

#define II_NEON(a, b, c, d, x, s, ac) { \
    a.val[0] = vaddq_u32(a.val[0], vaddq_u32(I_NEON(b, c, d).val[0], vaddq_u32((x).val[0], vdupq_n_u32(ac)))); \
    a.val[0] = ROTATELEFT_NEON(a, s).val[0]; \
    a.val[0] = vaddq_u32(a.val[0], b.val[0]); \
    a.val[1] = vaddq_u32(a.val[1], vaddq_u32(I_NEON(b, c, d).val[1], vaddq_u32((x).val[1], vdupq_n_u32(ac)))); \
    a.val[1] = ROTATELEFT_NEON(a, s).val[1]; \
    a.val[1] = vaddq_u32(a.val[1], b.val[1]); \
}


void MD5Hash_SIMD(array<std::string,8> input, bit32 state[8][4]);