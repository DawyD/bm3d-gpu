/*
 * Based on dct8x8_kernel2.cu provided in CUDA samples form NVIDIA Corporation.
 *
 * Provide functions to compute many 2D DCT and 2D IDCT of size 8x8 
*/


#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.  
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.  
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.  
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.  
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.  
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.  


/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

#define BLOCK_SIZE          8

/**
*  Width of macro-block
*/
#define KER2_BLOCK_WIDTH          128


/**
*  Height of macro-block
*/
#define KER2_BLOCK_HEIGHT         8


/**
*  Stride of shared memory buffer (2nd kernel)
*/
#define KER2_SMEMBLOCK_STRIDE     (KER2_BLOCK_WIDTH+1)


/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void InplaceDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float X07P = (*Vect0) + (*Vect7);
    float X16P = (*Vect1) + (*Vect6);
    float X25P = (*Vect2) + (*Vect5);
    float X34P = (*Vect3) + (*Vect4);

    float X07M = (*Vect0) - (*Vect7);
    float X61M = (*Vect6) - (*Vect1);
    float X25M = (*Vect2) - (*Vect5);
    float X43M = (*Vect4) - (*Vect3);

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    (*Vect0) = C_norm * (X07P34PP + X16P25PP);
    (*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    (*Vect4) = C_norm * (X07P34PP - X16P25PP);
    (*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    (*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    (*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    (*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    (*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}


/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0          [IN/OUT] - Pointer to the first element of vector
* \param Step           [IN/OUT] - Value to add to ptr to access other elements
*
* \return None
*/
__device__ void InplaceIDCTvector(float *Vect0, int Step)
{
    float *Vect1 = Vect0 + Step;
    float *Vect2 = Vect1 + Step;
    float *Vect3 = Vect2 + Step;
    float *Vect4 = Vect3 + Step;
    float *Vect5 = Vect4 + Step;
    float *Vect6 = Vect5 + Step;
    float *Vect7 = Vect6 + Step;

    float Y04P   = (*Vect0) + (*Vect4);
    float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
    float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

    float Y04M   = (*Vect0) - (*Vect4);
    float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
    float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

    (*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    (*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    (*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    (*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    (*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    (*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    (*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    (*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}


/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void DCT2D8x8(float *dst, const float *src, const uint size)
{
	__shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

	if (blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (threadIdx.y+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

	int offset = threadIdx.y * (BLOCK_SIZE*BLOCK_SIZE) + threadIdx.x;

	//Get macro-block address
	src += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;
	dst += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH;
	
	//8x1 blocks in one macro-block (threadIdx.y - index of block inside the macro-block)
	//Get the first element of the column in the block with index threadIdx.y
	src += offset;
	dst += offset;
	
	float *bl_ptr = block + offset;

#pragma unroll

	for (unsigned int i = 0; i < BLOCK_SIZE; i++)
		bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE]; //Load column to the shared mem

	//process rows
	InplaceDCTvector(bl_ptr - threadIdx.x + BLOCK_SIZE * threadIdx.x, 1);

	//process columns
	InplaceDCTvector(bl_ptr, BLOCK_SIZE);

	for (unsigned int i = 0; i < BLOCK_SIZE; i++)
		dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}


/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that
*  utilizes maximum warps capacity, assuming that it is enough of 8 threads
*  per block8x8.
*
* \param SrcDst                     [OUT] - Coefficients plane
* \param ImgStride                  [IN] - Stride of SrcDst
*
* \return None
*/

__global__ void IDCT2D8x8(float *dst, const float *src, const uint size)
{
	__shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

	if (blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH + (threadIdx.y+1) * BLOCK_SIZE*BLOCK_SIZE-1 >= size) return;

	int offset = threadIdx.y * (BLOCK_SIZE*BLOCK_SIZE) + threadIdx.x;
	
	src += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH; 
	dst += blockIdx.x * KER2_BLOCK_HEIGHT * KER2_BLOCK_WIDTH; 
	
	src += offset;
	dst += offset;
	
	float *bl_ptr = block + offset;

#pragma unroll

	for (unsigned int i = 0; i < BLOCK_SIZE; i++)
		bl_ptr[i * BLOCK_SIZE] = src[i * BLOCK_SIZE];

	//process rows
	InplaceIDCTvector(bl_ptr - threadIdx.x + BLOCK_SIZE * threadIdx.x, 1);

	//process columns
    	InplaceIDCTvector(bl_ptr, BLOCK_SIZE);    

	for (unsigned int i = 0; i < BLOCK_SIZE; i++)
		dst[i * BLOCK_SIZE] = bl_ptr[i * BLOCK_SIZE];
}

extern "C" void run_DCT2D8x8(	
	float *transformed_stacks,
	const float *gathered_stacks,
	const uint size,
	const dim3 num_threads,
	const dim3 num_blocks)
{
	DCT2D8x8<<<num_blocks, num_threads>>>(transformed_stacks, gathered_stacks, size);
}

extern "C" void run_IDCT2D8x8(
	float *gathered_stacks,
	const float *transformed_stacks,
	const uint size,
	const dim3 num_threads,
	const dim3 num_blocks)
{
	IDCT2D8x8<<<num_blocks, num_threads>>>(gathered_stacks, transformed_stacks, size);
}
