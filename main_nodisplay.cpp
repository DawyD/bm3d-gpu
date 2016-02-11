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
		std::cerr << "Usage: " << argv[0] << " NosiyImage DenoisedImage sigma [color [twostep [quiet]]]" << std::endl;
		return 1;
	}
	float sigma = strtof(argv[3],NULL);
	std::cout << "Sigma = " << sigma << std::endl;

	unsigned int channels = 1;
	if (argc >= 5 && strcmp(argv[4],"color") == 0)
	{
		std::cout << "Color image!" << std::endl;
		channels = 3;
	}
	bool twostep = false;
	if (argc >= 6 && strcmp(argv[5],"twostep") == 0)
	{
		std::cout << "Two step!" << std::endl;
		twostep = true;
	}
	bool verbose = true;
	if (argc >= 7 && strcmp(argv[6],"quiet") == 0)
	{
		verbose = false;
	}

	
	//Allocate images
	CImg<unsigned char> image(argv[1]);
	CImg<unsigned char> image2(image.width(), image.height(), 1, channels, 0);
	
	//Convert color image to YCbCr color space
	if (channels == 3)
		image = image.get_channels(0,2).RGBtoYCbCr();

	// Check for invalid input
	if(! image.data() )							
	{
		std::cerr << "Could not open or find the image" << std::endl;
		return 1;
	}

	std::cout << "width: " << image.width() << " height: " << image.height() << std::endl;

	//Launch BM3D
	try {
		BM3D bm3d;
		//		    (n, k,N, T,   p,sigma, L3D)
		bm3d.set_hard_params(19,8,16,2500,3,sigma, 2.7f);
		bm3d.set_wien_params(19,8,32,400, 3,sigma);
		bm3d.denoise_host_image(image.data(),
				 image2.data(),
				 image.width(),
				 image.height(),
				 channels,
				 twostep,
				 verbose);
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

    return 0;
}
