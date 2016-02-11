#ifndef _PARAMS_HPP_
#define _PARAMS_HPP_

struct Params
{
	/*
	RESTRICTIONS:
		k must be divisible by p
	*/
	unsigned int n;		// Area in which the similar blocks are searched
	unsigned int k;		// width and height of a patch
	unsigned int N;		// Maximal number of similar blocks in stack (without reference block)
	unsigned int T;		// Distance treshold under which two blocks are assumed simialr //DEV: NOT NECESSARY
	unsigned int Tn;	// Distance treshold under which two blocks are assumed simialr (with normalization facotr)
	unsigned int p;		// Step between reference patches
	float sigmap2;		//Expexted noise variance
	float L3Ds; 		// Treshold in colaborative filtering under which coefficients are replaced by zeros. -> for first step multiplied by sigma
	

	Params(unsigned int n = 32,
		   unsigned int k = 8,
		   unsigned int N = 8,
		   unsigned int T = 2500,
		   unsigned int p = 3,
		   float sigma = 10,
		   float L3D = 2.7f) : 
		n(n), k(k), N(N-1), T(T), Tn(T*k*k), p(p), sigmap2(sigma*sigma), L3Ds(L3D*sigma)  {}
};

#endif
