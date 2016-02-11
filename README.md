bm3d-gpu
========

CUDA-accelerated implementation of BM3D image denoising method

Author    : David Honzátko <honzatko@ksi.mff.cuni.cz>

# UNIX/LINUX USER GUIDE

The code is compilable on Unix/Linux. 

- Compilation. 
Automated compilation requires the make program.

- Libraries. 
This code requires the CUDA toolkit installed.

- Image format. 
All the image formats supported by the Cimg library.
For users that have convert or gm installed, it supports most of the image formats. Otherwise we recommend to use the .bmp format.


Usage:

1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linux).
Run 

    make

3. Run CUDA-accelerated BM3D image denoising application

    ./bm3d

The generic way to run the code is:

    ./bm3d noisyImage.bmp denoisedImage.bmp sigma [color [twostep [quiet]]]

Options:
- color - color image denoising
- twostep - process both steps of the BM3D method
- quiet - no information about the state of processing is displayed

Example of gray-scale denoising by the fisrt step of BM3D:

    ./bm3d lena_20.png lena_den.png 20
Example of color denoising by both steps of BM3D:

    ./bm3d lena_20_color.png lena_den_color.png 20 color twostep
Example of grayscale denoising by both steps of BM3D:

    ./bm3d lena_25.png lena_den.png 25 nocolor twostep
