#ifndef _INDICES_CUH_
#define _INDICES_CUH_

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

//Index handling
#define idx2(x,y,dim_x) ( (x) + ((y)*(dim_x)) )
#define idx3(x,y,z,dim_x,dim_y) ( (x) + ((y)*(dim_x)) + ((z)*(dim_x)*(dim_y)) )

template<typename T>
__device__ __forceinline__ T* idx2p(T* BaseAddress, uint Column, uint Row, uint pitch)
{
	return (T*)((char*)BaseAddress + Row * pitch) + Column;
}

struct uint2float1
{
	short x;
	short y;
	float val;

	__host__ __device__ uint2float1(short x, short y, float val) : x(x), y(y), val(val) { }
};


#endif