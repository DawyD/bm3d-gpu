#define NOMINMAX
#include "params.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm> //min  max
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>
#include <math.h>

#include "indices.cuh"

//2DDCT - has to be consistent with dct8x8.cu
#define KER2_BLOCK_WIDTH          128

//Exception handling
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>
#include <stdexcept>


//Debug
#include "stopwatch.hpp"
#include <fstream>
#include <iostream>

//Extern kernels

extern "C" void run_block_matching(
	const  uchar* __restrict image,
	uint2* stacks,
	uint* num_patches_in_stack,
	const uint2 image_dim,
	const uint2 stacks_dim,
	const Params params,
	const uint2 start_point,
	const dim3 num_threads,
	const dim3 num_blocks,
	const uint shared_memory_size
);

extern "C" void run_get_block(
	const uint2 start_point,
	const uchar* __restrict image,
	const uint2* __restrict stacks,
	const uint* __restrict num_patches_in_stack,
	float* patch_stack,
	const uint2 image_dim,
	const uint2 stacks_dim,
	const Params params,
	const dim3 num_threads,
	const dim3 num_blocks
);

extern "C" void run_DCT2D8x8(
	float *d_transformed_stacks,
	const float *d_gathered_stacks,
	const uint size,
	const dim3 num_threads,
	const dim3 num_blocks
);

extern "C" void run_hard_treshold_block(
	const uint2 start_point,
	float* patch_stack,
	float* w_P,
	const uint* __restrict num_patches_in_stack,
	const uint2 stacks_dim,
	const Params params,
	const dim3 num_threads,
	const dim3 num_blocks,
	const uint shared_memory_size
);

extern "C" void run_IDCT2D8x8(
	float *d_gathered_stacks,
	const float *d_transformed_stacks,
	const uint size,
	const dim3 num_threads,
	const dim3 num_blocks
);

extern "C" void run_aggregate_block(
	const uint2 start_point,
	const float* __restrict patch_stack,	
	const float* __restrict w_P,
	const uint2* __restrict stacks,
	const float* __restrict kaiser_window,
	float* numerator,
	float* denominator,
	const uint* __restrict num_patches_in_stack,
	const uint2 image_dim,
	const uint2 stacks_dim,
	const Params params,
	const dim3 num_threads,
	const dim3 num_blocks
);

extern "C" void run_aggregate_final(
	const float* __restrict numerator,
	const float* __restrict denominator,
	const uint2 image_dim,
	uchar* denoised_noisy_image,
	const dim3 num_threads,
	const dim3 num_blocks
);

extern "C" void run_wiener_filtering(
	const uint2 start_point,
	float* patch_stack,
	const float* __restrict patch_stack_basic,
	float*  w_P,
	const uint* __restrict num_patches_in_stack,
	uint2 stacks_dim,
	const Params params,
	const dim3 num_threads,
	const dim3 num_blocks,
	const uint shared_memory_size
);

//Cuda error handling
//Sometimes does not work
#define cuda_error_check(ans) { throw_on_cuda_error((ans),__FILE__, __LINE__); }
void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess)
	{
		std::stringstream ss;
		ss << file << "(" << line << "): " << cudaGetErrorString(code);
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}

class BM3D
{
private:
	//Image (vector of image channels)
	std::vector<uchar*> d_noisy_image;
	std::vector<uchar*> d_denoised_image;
	
	//Auxiliary arrays
	uint2* d_stacks;					//Addresses of similar patches to each reference patch of a batch
	std::vector<float*> d_numerator;	//Numerator used for aggregation
	std::vector<float*> d_denominator;	//Denminator used for aggregation
	uint* d_num_patches_in_stack;		//Number of similar patches for each referenca patch of a batch that are stored in d_stacks
	float* d_gathered_stacks;			//3D groups of a batch
	float* d_gathered_stacks_basic; 	//Only for two step denoising, contains wiener coefficients
	float* d_w_P;						//Weights for aggregation
	float* d_kaiser_window;				//Kaiser window used for aggregation


	//Reserved sizes
	int h_reserved_width;
	int h_reserved_height;
	int h_reserved_channels;
	bool h_reserved_two_step;
	uint2 h_batch_size; 				//h_batch_size.x has to be divisible by properties.warpSize

	//Denoising parameters
	Params h_hard_params;
	Params h_wien_params;

	//Device properties
	cudaDeviceProp properties;

	bool _verbose;
	
	//Allocate device buffers dependent on denoising parameters
	void allocate_device_auxiliary_arrays()
	{
		int maxk = std::max(h_wien_params.k,h_hard_params.k);
		int maxN = std::max(h_wien_params.N,h_hard_params.N);

		cuda_error_check( cudaMalloc((void**)&d_stacks, sizeof(uint2) * h_batch_size.x * h_batch_size.y * maxN) );

		cuda_error_check( cudaMalloc((void**)&d_num_patches_in_stack, sizeof(uint) * h_batch_size.x * h_batch_size.y ) );

		cuda_error_check( cudaMalloc((void**)&d_gathered_stacks, sizeof(float)*(maxN+1)*maxk*maxk*h_batch_size.x*h_batch_size.y) );
		
		cuda_error_check( cudaMalloc((void**)&d_w_P, sizeof(float) * h_batch_size.x*h_batch_size.y) );
		
		cuda_error_check( cudaMalloc((void**)&d_kaiser_window, sizeof(float) * maxk * maxk) );

		if (h_reserved_two_step)
			cuda_error_check( cudaMalloc((void**)&d_gathered_stacks_basic, sizeof(float)*(maxN+1)*maxk*maxk*h_batch_size.x*h_batch_size.y) );
	}

	//Allocate device buffers dependent on image dimensions
	void allocate_device_image(uint width, uint height, uint channels)
	{
		d_noisy_image.resize(channels);
		d_denoised_image.resize(channels);
		d_numerator.resize(channels);
		d_denominator.resize(channels);

		int size = width * height;
		for(auto & it : d_noisy_image) {
			cuda_error_check( cudaMalloc((void**)&it, sizeof(uchar) * size) );
		}

		for(auto & it : d_denoised_image) {
			cuda_error_check( cudaMalloc((void**)&it, sizeof(uchar) * size) );
		}

		for(auto & it : d_numerator) {
			cuda_error_check( cudaMalloc((void**)&it, sizeof(float) * size) );
		}

		for(auto & it : d_denominator) {
			cuda_error_check( cudaMalloc((void**)&it, sizeof(float) * size) );
		}

	}

	//Creates an kaiser window (only for k = 8, alpha = 2.0) and copies it to the device.
	void prepare_kaiser_window(uint k)
	{
		std::vector<float> kaiserWindow(k*k);
		if (k == 8)
		{
			//! First quarter of the matrix
			kaiserWindow[0 + k * 0] = 0.1924f; kaiserWindow[0 + k * 1] = 0.2989f; kaiserWindow[0 + k * 2] = 0.3846f; kaiserWindow[0 + k * 3] = 0.4325f;
			kaiserWindow[1 + k * 0] = 0.2989f; kaiserWindow[1 + k * 1] = 0.4642f; kaiserWindow[1 + k * 2] = 0.5974f; kaiserWindow[1 + k * 3] = 0.6717f;
			kaiserWindow[2 + k * 0] = 0.3846f; kaiserWindow[2 + k * 1] = 0.5974f; kaiserWindow[2 + k * 2] = 0.7688f; kaiserWindow[2 + k * 3] = 0.8644f;
			kaiserWindow[3 + k * 0] = 0.4325f; kaiserWindow[3 + k * 1] = 0.6717f; kaiserWindow[3 + k * 2] = 0.8644f; kaiserWindow[3 + k * 3] = 0.9718f;

			//! Completing the rest of the matrix by symmetry
			for(unsigned i = 0; i < k / 2; i++)
				for (unsigned j = k / 2; j < k; j++)
					kaiserWindow[i + k * j] = kaiserWindow[i + k * (k - j - 1)];

			for (unsigned i = k / 2; i < k; i++)
				for (unsigned j = 0; j < k; j++)
					kaiserWindow[i + k * j] = kaiserWindow[k - i - 1 + k * j];
		} 
		else
		        for (unsigned i = 0; i < k * k; i++)
            			kaiserWindow[i] = 1.0f;

		cuda_error_check( cudaMemcpy(d_kaiser_window,&kaiserWindow[0],k*k*sizeof(float),cudaMemcpyHostToDevice));
	}
	
	//Copy image to device
	void copy_device_image(const uchar * src_image, int width, int height, int channels)
	{
		size_t image_size = width * height;
		for(int i = 0; i < channels; ++i) {
			//Copy image to device
			cuda_error_check( cudaMemcpy(d_noisy_image[i],src_image+i*image_size,image_size*sizeof(uchar),cudaMemcpyHostToDevice));
		}
	}
	
	//Compute launch parameters for block-matching kernel
	void get_BM_launch_parameters(
		const Params & params, 	//IN: Denoising parameters
		dim3 & num_threads,		//OUT: number of threads
		dim3 & num_blocks,		//OUT: numbe of blocks
		uint & s_mem_size)		//OUT: shared memory size
	{
		//Determine number of warps form block-matching according to the size of shared memory. 
		const uint p_block_width = (properties.warpSize * params.p) + params.k - 1;
		const uint s_image_p_size = p_block_width * params.k * sizeof(uchar);

		const float shared_mem_usage = 1.0f; // 0 - 1
		const uint shared_mem_avaliable = (uint)(properties.sharedMemPerBlock * shared_mem_usage) - s_image_p_size;

		//Block-matching shared memory sizes per warp
		const uint s_diff_size = p_block_width * sizeof(float);
		const uint s_patches_in_stack_size = properties.warpSize * sizeof(uint);
		const uint s_patch_stacks_size = params.N * properties.warpSize * sizeof(uint2float1);

		const uint num_warps = std::min(shared_mem_avaliable / (s_diff_size + s_patches_in_stack_size + s_patch_stacks_size),32u);
		if (_verbose)
			std::cout << "Number of warps: " << num_warps << std::endl;

		//Block-matching Launch parameters
		s_mem_size = ((s_diff_size + s_patches_in_stack_size + s_patch_stacks_size) * num_warps) + s_image_p_size;		
		num_threads = dim3(properties.warpSize*num_warps, 1);
		num_blocks = dim3(h_batch_size.x / properties.warpSize, h_batch_size.y);
	}

	/*
	Launch first step of BM3D. It produces basic estimate in denoised_image arrays.
	*/
	void first_step(std::vector<uchar*> & denoised_image, int width, int height, int channels)
	{	
		//image dimensions
		const uint2 image_dim = make_uint2(width,height);

		//dimensions limiting addresses of reference patches
		const uint2 stacks_dim = make_uint2(width - (h_hard_params.k - 1), height - (h_hard_params.k - 1));

		int paramN1 = h_hard_params.N + 1; //maximal size of a stack with a reference patch

		//Determine launch parameteres for block-matching kernel
		dim3 num_threads_bm;
		dim3 num_blocks_bm;
		uint s_size_bm;
		get_BM_launch_parameters(h_hard_params, num_threads_bm, num_blocks_bm, s_size_bm);
		
		//Determine launch parameteres for get and aggregate kernels
		const dim3 num_threads(h_hard_params.k, h_hard_params.k);
		const dim3 num_blocks(h_batch_size.x, h_batch_size.y);

		//Determine launch parameteres for DCT kernel
		const uint trans_size = h_hard_params.k*h_hard_params.k*paramN1*h_batch_size.x*h_batch_size.y;
		const dim3 num_blocks_tr((trans_size + (KER2_BLOCK_WIDTH*h_hard_params.k) - 1) / (KER2_BLOCK_WIDTH*h_hard_params.k), 1, 1);
		const dim3 num_threads_tr(h_hard_params.k, KER2_BLOCK_WIDTH/h_hard_params.k, 1);

		//Determine launch parameteres for filtering kernel
		const uint s_size_t = h_hard_params.k*h_hard_params.k*(paramN1+1)*sizeof(float); //+1 for avoinding bank conflicts

		//Determine launch parameteres for final division kernel
		const dim3 num_threads_f(128, 4);
		const dim3 num_blocks_f((width + num_threads_f.x - 1) / num_threads_f.x, (height + num_threads_f.y - 1) / num_threads_f.y);
		
		//Create and copy to device kaiser window
		prepare_kaiser_window(h_hard_params.k);
		
		//Timers
		Stopwatch time_blockmatching;
		Stopwatch time_get;
		Stopwatch time_transform;
		Stopwatch time_itransform;
		Stopwatch time_aggregate;
		Stopwatch time_treshold;
		
		
		//Batch processing: in each iteration only the batch_size reference patches are processed. 
		uint2 start_point;
		for(start_point.y = 0; start_point.y < stacks_dim.y + h_hard_params.p - 1; start_point.y+=(h_batch_size.y*h_hard_params.p))
		{
			//Show progress
			if (_verbose)
			{
				int percent = (int)(((float)start_point.y / (float)stacks_dim.y) * (float)100);
				std::cout << "\rProcessing " << percent << "%" << std::flush;
			}
			for(start_point.x = 0; start_point.x < stacks_dim.x + h_hard_params.p - 1; start_point.x+=(h_batch_size.x*h_hard_params.p))
			{
				if (_verbose)
					time_blockmatching.start();
				
				//Finds similar patches for each reference patch of a batch and stores them in d_stacks array
				run_block_matching(
					d_noisy_image[0],			// IN: Image	
					d_stacks,					// OUT: Array of adresses of similar patches
					d_num_patches_in_stack,		// OUT: Array containing numbers of these addresses
					image_dim,					// IN: Image dimensions
					stacks_dim,					// IN: Dimensions limiting addresses of reference patches
					h_hard_params,				// IN: Denoising parameters 
					start_point,				// IN: Address of the top-left reference patch of a batch
					num_threads_bm,				// CUDA: Threads in block 
					num_blocks_bm,				// CUDA: Blocks in grid
					s_size_bm					// CUDA: Shared memory size
				);

				cuda_error_check( cudaGetLastError() );
				cuda_error_check( cudaDeviceSynchronize() );

				if (_verbose)
					time_blockmatching.stop();

				for (int channel = 0; channel < channels; ++channel)
				{
					if (_verbose)
						time_get.start();

					//Assembles 3D groups of a batch according to the d_stacks array
					run_get_block(
						start_point,				//IN: First reference patch of a batch
						d_noisy_image[channel],		//IN: Image
						d_stacks,					//IN: Array of adresses of similar patches
						d_num_patches_in_stack,		//IN: Numbers of patches in 3D groups
						d_gathered_stacks,			//OUT: Assembled 3D groups
						image_dim,					//IN: Image dimensions
						stacks_dim,					//IN: Dimensions limiting addresses of reference patches
						h_hard_params,				//IN: Denoising parameters
						num_threads,				//CUDA: Threads in block
						num_blocks					//CUDA: Blocks in grid
					);
			
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
					if (_verbose)
					{
						time_get.stop();
						time_transform.start();
					}
					
					//Apply the 2D DCT transform to each layer of 3D group
					run_DCT2D8x8(d_gathered_stacks, d_gathered_stacks, trans_size, num_threads_tr, num_blocks_tr);
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );


					if (_verbose)
					{
						time_transform.stop();
						time_treshold.start();
					}

					
					/*
					1) 1D Walsh-Hadamard transform of proper size on the 3rd dimension of each 3D group of a batch to complete the 3D transform.
					2) Hard thresholding
					3) Inverse 1D Walsh-Hadamard trannsform.
					4) Compute the weingt of each 3D group
					*/
					
					run_hard_treshold_block(
						start_point,			//IN: First reference patch of a batch
						d_gathered_stacks,	//IN/OUT: 3D groups with thransfomed patches
						d_w_P, 					//OUT: Weight of each 3D group
						d_num_patches_in_stack,	//IN: Numbers of patches in 3D groups
						stacks_dim,				//IN: Dimensions limiting addresses of reference patches
						h_hard_params,			//IN: Denoising parameters
						num_threads,			//CUDA: Threads in block
						num_blocks,				//CUDA: Blocks in grid
						s_size_t				//CUDA: Shared memory size
					);
					
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
					if (_verbose)
					{
						time_treshold.stop();
						time_itransform.start();
					}
					
					//Apply inverse 2D DCT transform to each layer of 3D group
					run_IDCT2D8x8(d_gathered_stacks, d_gathered_stacks, trans_size, num_threads_tr, num_blocks_tr);
					
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
					if (_verbose)
					{
						time_itransform.stop();
						time_aggregate.start();
					}
				
					//Aggregates filtered patches of all 3D groups of a batch into numerator and denominator buffers
					run_aggregate_block(
						start_point,				//IN: First reference patch of a batch
						d_gathered_stacks,			//IN: 3D groups with thransfomed patches
						d_w_P,						//IN: Numbers of non zero coeficients after 3D thresholding
						d_stacks,					//IN: Array of adresses of similar patches
						d_kaiser_window,			//IN: Kaiser window
						d_numerator[channel],		//IN/OUT: Numerator aggregation buffer
						d_denominator[channel],		//IN/OUT: Denominator aggregation buffer
						d_num_patches_in_stack,		//IN: Numbers of patches in 3D groups
						image_dim,					//IN: Image dimensions
						stacks_dim,					//IN: Dimensions limiting addresses of reference patches
						h_hard_params,				//IN: Denoising parameters
						num_threads,				//CUDA: Threads in block
						num_blocks					//CUDA: Blocks in grid
					);
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
			
			
					if (_verbose)
						time_aggregate.stop();
				}
			}	
		}	

		//Divide numerator by denominator and save resullt to output image
		for (int channel = 0; channel < channels; ++channel)
		{
			run_aggregate_final(
				d_numerator[channel],			//IN: Numerator aggregation buffer
				d_denominator[channel],			//IN: Denominator aggregation buffer
				image_dim,						//IN: Image dimensions
				denoised_image[channel],		//OUT: Image estimate
				num_threads_f,					//CUDA: Threads in block
				num_blocks_f 					//CUDA: Blocks in grid
			);
			cuda_error_check( cudaGetLastError() );
			cuda_error_check( cudaDeviceSynchronize() );
		}

		if(_verbose)
		{
			//Print timers
			std::cout << "\rFirst step details:" << std::endl; //DEBUG: erase line
			std::cout << "  Block-Matching took: " << time_blockmatching.getSeconds() << std::endl;
			std::cout << "  Get took: " << time_get.getSeconds() << std::endl;
			std::cout << "  Transform took: " << time_transform.getSeconds() << std::endl;
			std::cout << "  Tresholding took: " << time_treshold.getSeconds() << std::endl;
			std::cout << "  Inverse transform took: " << time_itransform.getSeconds() << std::endl;
			std::cout << "  Aggregation took: " << time_aggregate.getSeconds() << std::endl;
		}

	}

	void second_step(std::vector<uchar*> & denoised_image, int width, int height, int channels)
	{	
		//Image dimensions
		const uint2 image_dim = make_uint2(width,height);

		//Dimensions limiting addresses of reference patches
		const uint2 stacks_dim = make_uint2(width - (h_wien_params.k - 1), height - (h_wien_params.k - 1));

		int paramN1 = h_wien_params.N + 1; //Maximal size of a stack with a reference patch

		//Determine launch parameteres for block-matching kernel
		dim3 num_threads_bm;
		dim3 num_blocks_bm;
		uint s_size_bm;
		get_BM_launch_parameters(h_wien_params, num_threads_bm, num_blocks_bm, s_size_bm);
		
		//Determine launch parameteres for get and aggregate kernels
		const dim3 num_threads(h_wien_params.k, h_wien_params.k);
		const dim3 num_blocks(h_batch_size.x, h_batch_size.y);
		
		//Determine launch parameteres for DCT kernel
		const uint trans_size = h_wien_params.k*h_wien_params.k*paramN1*h_batch_size.x*h_batch_size.y;
		const dim3 num_blocks_tr((trans_size + (KER2_BLOCK_WIDTH*h_wien_params.k) - 1) / (KER2_BLOCK_WIDTH*h_wien_params.k), 1, 1); 
		const dim3 num_threads_tr(h_wien_params.k, KER2_BLOCK_WIDTH/h_wien_params.k, 1);

		//Determine launch parameteres for filtering kernel
		const uint s_size_t = 2*h_wien_params.k*h_wien_params.k*(paramN1+1)*sizeof(float); //+1 for avoinding bank conflicts

		//Determine launch parameteres for final division kernel
		const dim3 num_threads_f(128, 4);
		const dim3 num_blocks_f((width + num_threads_f.x - 1) / num_threads_f.x, (height + num_threads_f.y - 1) / num_threads_f.y);

		//Create and copy to device kaiser window
		prepare_kaiser_window(h_wien_params.k);
		
		//Timers
		Stopwatch time_blockmatching;
		Stopwatch time_get;
		Stopwatch time_get2;
		Stopwatch time_transform;
		Stopwatch time_transform2;
		Stopwatch time_itransform;
		Stopwatch time_aggregate;
		Stopwatch time_wien;
		Stopwatch time_times_wien;
		
		
		uint2 start_point;

		for(start_point.y = 0; start_point.y < stacks_dim.y + h_wien_params.p - 1; start_point.y+=(h_batch_size.y*h_wien_params.p))
		{
			//Show progress
			if (_verbose)
			{
				int percent = (int)(((float)start_point.y / (float)stacks_dim.y) * (float)100);
				std::cout << "\rProcessing " << percent << "%" << std::flush;
			}
			for(start_point.x = 0; start_point.x < stacks_dim.x + h_wien_params.p - 1; start_point.x+=(h_batch_size.x*h_wien_params.p))
			{
				if (_verbose)
					time_blockmatching.start();

				run_block_matching(
					d_denoised_image[0],	// IN: Image	
					d_stacks,				// OUT: Array of adresses of similar patches
					d_num_patches_in_stack,	// OUT: Number of blocks on each adress
					image_dim,				// IN: Image dimensions
					stacks_dim,				// IN: Image_m dimensions
					h_wien_params,			// IN: Parameters 
					start_point,			// IN: Line to process
					num_threads_bm,			// CUDA: Threads in block 
					num_blocks_bm,			// CUDA: Blocks in grid
					s_size_bm				// CUDA: Shared memory size
				);
				cuda_error_check( cudaGetLastError() );
				cuda_error_check( cudaDeviceSynchronize() );


				if (_verbose)
					time_blockmatching.stop();

				for (int channel = 0; channel < channels; ++channel)
				{
					if (_verbose)
						time_get.start();
				
					//Get patches from basic image estimate to 3D auxiliary array according to the addresess form block-matching
					run_get_block(
						start_point,				//IN: First reference patch of a batch
						d_denoised_image[channel], 	//IN: Basic image estimate (produced by 1st step)
						d_stacks,					//IN: Array of adresses of similar patches
						d_num_patches_in_stack,		//IN: Numbers of patches in 3D groups
						d_gathered_stacks_basic,	//OUT: Assembled 3D groups
						image_dim,					//IN: Image dimensions
						stacks_dim,					//IN: Dimensions limiting addresses of reference patches
						h_wien_params,				//IN: Denoising parameters
						num_threads,				//CUDA: Threads in block
						num_blocks					//CUDA: Blocks in grid
					);
					
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
					//Get patches from noisy image to 3D auxiliary array according to the addresess form block-matching
					run_get_block(
						start_point,				//IN: First reference patch of a batch
						d_noisy_image[channel],		//IN: Basic image estimate (produced by 1st step)
						d_stacks,					//IN: Array of adresses of similar patches
						d_num_patches_in_stack,		//IN: Numbers of patches in 3D groups
						d_gathered_stacks,			//OUT: Assembled 3D groups
						image_dim,					//IN: Image dimensions
						stacks_dim,					//IN: Dimensions limiting addresses of reference patches
						h_wien_params,				//IN: Denoising parameters
						num_threads,				//CUDA: Threads in block
						num_blocks					//CUDA: Blocks in grid
					);
				
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );

				
					if (_verbose)
					{
						time_get.stop();
						time_transform.start();
					}
		
					//Apply 2D DCT transform to each layer of 3D group that contains noisy patches
					run_DCT2D8x8(d_gathered_stacks, d_gathered_stacks, trans_size, num_threads_tr, num_blocks_tr);
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
					//Apply 2D DCT transform to each layer of 3D group that contains patches from basic image estimate
					run_DCT2D8x8(d_gathered_stacks_basic, d_gathered_stacks_basic, trans_size, num_threads_tr, num_blocks_tr);
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
		
					
					if (_verbose)
					{
						time_transform.stop();
						time_wien.start();
					}
		
					/*
					1) 1D Walsh-Hadamard transform of proper size on the 3rd dimension of each 3D noisy group (from noisy patches) and each 3D basic group (pathes from the basic image estimate)
					2) Compute wiener coeficients from basic groups
					3) Filtering: Element-wise multiplication between noisy group and corresponding wiener coefficinets
					4) Inverse 1D transform to the filtered groups
					5) Compute the weingt of each 3D group
					*/
					run_wiener_filtering(
						start_point,				//IN: First reference patch of a batch
						d_gathered_stacks,			//IN/OUT: 3D groups with thransfomed noisy patches that will be filtered
						d_gathered_stacks_basic,	//IN: 3D groups with thransfomed basic patches estimates
						d_w_P,						//OUT: Weight of each 3D group
						d_num_patches_in_stack,		//IN: Numbers of patches in 3D groups
						stacks_dim,					//IN: Dimensions limiting addresses of reference patches
						h_wien_params,				//IN: Denoising parameters
						num_threads,				//CUDA: Threads in block
						num_blocks,					//CUDA: Blocks in grid
						s_size_t					//CUDA: Shared memory size
					);

					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );

				
					if (_verbose)
					{
						time_wien.stop();
						time_itransform.start();
					}
					
					//Apply 2D IDCT transform to each layer of 3D group that contains filtered patches
					run_IDCT2D8x8(d_gathered_stacks, d_gathered_stacks, trans_size, num_threads_tr, num_blocks_tr);
					
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
					
				
					if (_verbose)
					{
						time_itransform.stop();
						time_aggregate.start();
					}


					//Aggregate filtered patches of all 3D groups of a batch into numerator and denominator buffers
					run_aggregate_block(
						start_point,			//IN: First reference patch of a batch
						d_gathered_stacks,		//IN: 3D groups with thransfomed patches
						d_w_P,					//IN: Numbers of non zero coeficients after 3D thresholding
						d_stacks,				//IN: Array of adresses of similar patches
						d_kaiser_window,		//IN: Kaiser window
						d_numerator[channel],	//IN/OUT: Numerator aggregation buffer
						d_denominator[channel],	//IN/OUT: Denominator aggregation buffer
						d_num_patches_in_stack,	//IN: Numbers of patches in 3D groups
						image_dim,				//IN: Image dimensions
						stacks_dim,				//IN: Dimensions limiting addresses of reference patches
						h_wien_params,			//IN: Denoising parameters
						num_threads,			//CUDA: Threads in block
						num_blocks				//CUDA: Blocks in grid
					);			
					cuda_error_check( cudaGetLastError() );
					cuda_error_check( cudaDeviceSynchronize() );
				
				
					if (_verbose)
						time_aggregate.stop();
				}	
			}
		}
		//Divide numerator by denominator and save resullt to output image
		for (int channel = 0; channel < channels; ++channel)
		{
			run_aggregate_final(
				d_numerator[channel],			//IN: Aggregation buffer
				d_denominator[channel],			//IN: Aggregation buffer
				image_dim,						//IN: Image dimensions
				denoised_image[channel],		//OUT: Image estimate
				num_threads_f,
				num_blocks_f 
			);
			cuda_error_check( cudaGetLastError() );
			cuda_error_check( cudaDeviceSynchronize() );
		}

		if(_verbose)
		{
			//Print timers
			std::cout << "\rSecond step details:" << std::endl;
			std::cout << "  BlockMatching took: " << time_blockmatching.getSeconds() << std::endl;
			std::cout << "  2x Get took: " << time_get.getSeconds() << std::endl;
			std::cout << "  2x Transform took: " << time_transform.getSeconds() << std::endl;
			std::cout << "  Wiener filtering took: " << time_wien.getSeconds() << std::endl;
			std::cout << "  Inverse transform took: " << time_itransform.getSeconds() << std::endl;
			std::cout << "  Aggregation took: " << time_aggregate.getSeconds() << std::endl;
		}
	}

	//Copy image from device to host
	void copy_host_image(uchar * dst_image, int width, int height, int channels)
	{
		size_t image_size = width * height;
		for (int channel = 0; channel < channels; ++channel)
		{
			cuda_error_check( cudaMemcpy(
						dst_image+channel*image_size,		// Destination
						d_denoised_image[channel],			// Source
						image_size*sizeof(uchar),			// Size
						cudaMemcpyDeviceToHost) );			// Copy direction
		}
	}

	//Free all buffers allocated on device that are dependent on 'denoising parameters'.
	void free_device_auxiliary_arrays()
	{
		cuda_error_check( cudaFree(d_stacks) );
		cuda_error_check( cudaFree(d_num_patches_in_stack) );

		cuda_error_check( cudaFree(d_gathered_stacks));
		cuda_error_check( cudaFree(d_w_P));

		cuda_error_check( cudaFree(d_kaiser_window));

		if (h_reserved_two_step)
			cuda_error_check( cudaFree(d_gathered_stacks_basic));
	}

	//Free all buffers allocated on device that are dependent on image dimensions
	void free_device_image()
	{
		for (auto & it : d_noisy_image)
			cuda_error_check( cudaFree(it) );
		d_noisy_image.clear();
		for (auto & it : d_denoised_image)
			cuda_error_check( cudaFree(it) );
		d_denoised_image.clear();
		for(auto & it : d_numerator) {
			cuda_error_check( cudaFree(it) );
		}
		d_numerator.clear();
		for(auto & it : d_denominator) {
			cuda_error_check( cudaFree(it) );
		}
		d_denominator.clear();
	}
	
	//Initialize necessary arrays for aggregation by value 0
	void null_aggregation_buffers(int width, int height)
	{
		int size = width * height;
		for(auto & it : d_numerator) {
			cuda_error_check( cudaMemset(it, 0, size * sizeof(float)) );
		}
		for(auto & it : d_denominator) {
			cuda_error_check( cudaMemset(it, 0, size * sizeof(float)) );
		}
	}
	
public:
	BM3D() : 
		h_hard_params(),
		h_wien_params(),
		d_gathered_stacks(0), d_gathered_stacks_basic(0), d_w_P(0), d_stacks(0), d_num_patches_in_stack(0),
		h_reserved_width(0), h_reserved_height(0), h_reserved_channels(0), h_reserved_two_step(0), d_kaiser_window(0), _verbose(false)
	{
		int device;
		cuda_error_check( cudaGetDevice(&device) );
		cuda_error_check( cudaGetDeviceProperties(&properties,device) );

		h_batch_size = make_uint2(256,128);
	}
	BM3D(uint n, uint k, uint N, uint T, uint p, float sigma, float L3D, bool seceon_step) : 
		h_hard_params(n, k, N, T, p, sigma, L3D),
		h_wien_params(n, k, N, T, p, sigma, L3D),
		d_gathered_stacks(0), d_gathered_stacks_basic(0), d_w_P(0), d_stacks(0), d_num_patches_in_stack(0),
		h_reserved_width(0), h_reserved_height(0), h_reserved_channels(0), h_reserved_two_step(0), d_kaiser_window(0), _verbose(false)
	{
		int device;
		cuda_error_check( cudaGetDevice(&device) );
		cuda_error_check( cudaGetDeviceProperties(&properties,device) );

		h_batch_size = make_uint2(256,128);
		
		if (k != 8) 
			throw std::invalid_argument("k has to be 8, other values not implemented yet.");
	}

	~BM3D()
	{
		free_device_image();
		free_device_auxiliary_arrays();
	}


	/*
	Source image is denoised unig BM3D algorithm
	src_image and dst_image are arrays allocated in the host memory and the pixels are stored here by the channels. 
	First width*height pixels represent luma (Y) component and each following width*height pixels represent color components
	*/
	void denoise_host_image(uchar *src_image, uchar *dst_image, int width, int height, int channels, bool two_step)
	{
		Stopwatch total;
		total.start();

		//Allocation
		if (h_reserved_width != width || h_reserved_height != height || h_reserved_channels != channels || h_reserved_two_step != two_step)
			reserve(width, height, channels, two_step);

		if (h_reserved_width == 0 || h_reserved_height == 0 || h_reserved_channels == 0 )
			return;

		Stopwatch p1;
		p1.start();
		
		//Copying
		copy_device_image(src_image, width, height, channels);

		//1st denoising step
		null_aggregation_buffers(width,height);
		first_step(d_denoised_image, width, height, channels);

		p1.stop();
		if (_verbose)
			std::cout << "1st step took: " << p1.getSeconds() << std::endl;
		
		//2nd denoising step
		if (two_step)
		{
			Stopwatch p2;
			p2.start();
			null_aggregation_buffers(width,height);
			second_step(d_denoised_image, width, height, channels);
			if (_verbose)
				std::cout << "2nd step took: " << p2.getSeconds() << std::endl;
		}

		//Copy back
		copy_host_image(dst_image, width, height, channels);

		if(_verbose)
			std::cout << "Total time: " << total.getSeconds() << std::endl;
	}

	/*void denoise_device_image(uchar *src_image, uchar *dst_image, int width, int height, int channels, bool two_step)
	{
		//TODO
	}*/

	void set_hard_params(uint n, uint k, uint N, uint T, uint p, float sigma, float L3D)
	{
		if (h_hard_params.k != k || h_hard_params.N != N)
		{
			h_hard_params = Params(n,k,N,T,p,sigma,L3D);
			free_device_auxiliary_arrays();
			allocate_device_auxiliary_arrays();
		}
		else
			h_hard_params = Params(n,k,N,T,p,sigma,L3D);
		
		if (k != 8) 
			throw std::invalid_argument("k has to be 8, other values not implemented yet.");
	}
	void set_wien_params(uint n, uint k, uint N, uint T, uint p, float sigma)
	{
		if (h_wien_params.k != k || h_wien_params.N != N){
			h_wien_params = Params(n,k,N,T,p,sigma,0.0);
			free_device_auxiliary_arrays();
			allocate_device_auxiliary_arrays();
		}
		else
			h_wien_params = Params(n,k,N,T,p,sigma,0.0);
		
		if (k != 8) 
			throw std::invalid_argument("k has to be 8, other values not implemented yet.");
	}

	void set_verbose(bool verbose)
	{
		_verbose = verbose;
	}

	void reserve(int width, int height, int channels, bool two_step)
	{
		h_reserved_width = width;
		h_reserved_height = height;
		h_reserved_channels = channels;
		h_reserved_two_step = two_step;

		free_device_image();
		free_device_auxiliary_arrays(); //TODO: not necessary

		allocate_device_image(width,height,channels);
		allocate_device_auxiliary_arrays(); //TODO: not necessary
	}
	void clear()
	{
		h_reserved_width = 0;
		h_reserved_height = 0;
		h_reserved_channels = 0;
		h_reserved_two_step = 0;

		free_device_image();
		free_device_auxiliary_arrays();
	}
};
