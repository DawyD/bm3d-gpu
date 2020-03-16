#include <iostream>
#include <string>

#include "bm3d.hpp"
#define cimg_display 0
#include "CImg.h"

using namespace cimg_library;

int main(int argc, char** argv)
{
	if( argc < 4 )
	{
		std::cerr << "Usage: " << argv[0] << " NosiyImage DenoisedImage sigma [color [twostep [quiet [ReferenceImage]]]]" << std::endl;
		return 1;
	}
	float sigma = strtof(argv[3],NULL);

	unsigned int channels = 1;
	if (argc >= 5 && strcmp(argv[4],"color") == 0)
	{
		channels = 3;
	}
	bool twostep = false;
	if (argc >= 6 && strcmp(argv[5],"twostep") == 0)
	{
		twostep = true;
	}
	bool verbose = true;
	if (argc >= 7 && strcmp(argv[6],"quiet") == 0)
	{
		verbose = false;
	}

	if (verbose)
	{
		std::cout << "Sigma = " << sigma << std::endl;
		if (twostep)
			std::cout << "Number of Steps: 2" << std::endl;
		else
			std::cout << "Number of Steps: 1" << std::endl;

		if (channels > 1)
			std::cout << "Color denoising: yes" << std::endl;
		else
			std::cout << "Color denoising: no" << std::endl;
	}
	
	//Allocate images
	CImg<unsigned char> image(argv[1]);
	CImg<unsigned char> image2(image.width(), image.height(), 1, channels, 0);
	std::vector<unsigned int> sigma2(channels);
	sigma2[0] = 25 * 25;
	
	//Convert color image to YCbCr color space
	if (channels == 3)
	{
		image = image.get_channels(0, 2).RGBtoYCbCr();
		//Convert the sigma^2 variance to the YCbCr color space
		long s = sigma * sigma;
		sigma2[0] = ((66l*66l*s + 129l*129l*s + 25l*25l*s) / (256l*256l));
		sigma2[1] = ((38l*38l*s + 74l*74l*s + 112l*112l*s) / (256l*256l)),
		sigma2[2] = ((112l*112l*s + 94l*94l*s + 18l*18l*s) / (256l*256l));
	}

	std::cout << "Noise variance for individual channels (YCrCb if color): ";
	for (unsigned int k = 0; k < sigma2.size(); k++)
		std::cout << sigma2[k] << " ";
	std::cout << std::endl;

	// Check for invalid input
	if(! image.data() )							
	{
		std::cerr << "Could not open or find the image" << std::endl;
		return 1;
	}

	if(verbose)
		std::cout << "width: " << image.width() << " height: " << image.height() << std::endl;

	//Launch BM3D
	try {
		BM3D bm3d;
		//		    (n, k,N, T,   p,sigma, L3D)
		bm3d.set_hard_params(19,8,16,2500,3, 2.7f);
		bm3d.set_wien_params(19,8,32,400, 3);
		bm3d.set_verbose(verbose);
		bm3d.denoise_host_image(
				image.data(),
				image2.data(),
				image.width(),
				image.height(),
				channels,
				sigma2.data(),
				twostep);
	}
	catch(std::exception & e)  {
		std::cerr << "There was an error while processing image: " << std::endl << e.what() << std::endl;
		return 1;
	}
	
	if (channels == 3) //color
		//Convert back to RGB color space
		image2 = image2.get_channels(0,2).YCbCrtoRGB();
	else
		image2 = image2.get_channel(0);
	//Save denoised image
	image2.save( argv[2] );

	if (argc >= 8)
	{
		CImg<unsigned char> reference_image(argv[7]);
		std::cout << "PSNR:" << reference_image.PSNR(image2) << std::endl;
	}

    return 0;
}
