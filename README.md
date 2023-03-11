BM3D-GPU
========

CUDA-accelerated implementation of BM3D image denoising method

Author    : David Honzátko <david.honzatko@epfl.ch>

# Unix/Linux User Guide

The code is compilable on Unix/Linux. 

- Compilation. 
Automated compilation requires the cmake program.

- Libraries. 
This code requires the CUDA toolkit installed.

- Image format. 
All the image formats supported by the Cimg library.
For users that have convert or gm installed, it supports most of the image formats. Otherwise we recommend to use the .bmp format.


Usage:

1. Download the code package and extract it. Go to that directory. 

2. Create build directory, create the makefiles using cmake and compile the application
Run 
```
mkdir build
cd build
cmake ..
make
```
3. Run CUDA-accelerated BM3D image denoising application
```
./bm3d
```
The generic way to run the code is:
```
./bm3d NoisyImage.png DenoisedImage.png sigma [color [twostep [quiet [ReferenceImage]]]]
```
Options:
- color - color image denoising (experimental only)
- twostep - process both steps of the BM3D method
- quiet - no information about the state of processing is displayed
- ReferenceImage - if provided, computes and prints PSNR between the ReferenceImage and DenoisedImage

Example of gray-scale denoising by the fisrt step of BM3D:
```
./bm3d lena_20.png lena_den.png 20
```
Example of color denoising by both steps of BM3D:
```
./bm3d lena_20_color.png lena_den_color.png 20 color twostep
```
Example of grayscale denoising by both steps of BM3D with PSNR computation:
```
./bm3d lena_25.png lena_den.png 25 nocolor twostep quiet lena.png
``` 
# Citation
If you find this implementation useful please cite the following paper in your work:

    @article{bm3d-gpu,
        author = {Honzátko, David and Kruliš, Martin},
        year = {2017}, month = {11},
        title = {Accelerating block-matching and 3D filtering method for image denoising on GPUs},
        booktitle = {Journal of Real-Time Image Processing}
    }
