CPP=g++
CFLAGS= -O2 --std=c++11
LDFLAGS=
LIBS=cufft cudart
LIBDIRS=/usr/local/cuda-7.5/lib64
HEADERS=$(shell find . -name '*.hpp')
CUHEADERS=$(shell find . -name '*.cuh')
INCLUDE=/usr/local/cuda-7.5/include
TARGET=bm3d

all: $(TARGET)

$(TARGET): main_nodisplay.o  filtering.o blockmatching.o dct8x8.o
	@echo Compilling and linking executable "$@" ...
	@nvcc -m64 $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) filtering.o blockmatching.o dct8x8.o $< -o $@

main_nodisplay.o: main_nodisplay.cpp $(HEADERS)
	@echo Compilling main_nodiplay.cpp
	$(CPP) $(CFLAGS) -m64 -c  $(addprefix -I,$(INCLUDE)) $< -o $@

filtering.o: filtering.cu indices.cuh params.hpp
	@echo Compilling filtering.cu
	@nvcc $(addprefix -I,$(INCLUDE)) -m64 -c -arch=sm_30 $< -o $@
blockmatching.o: blockmatching.cu indices.cuh params.hpp
	@echo Compilling blockmatching.cu
	@nvcc $(addprefix -I,$(INCLUDE)) -m64 -c -arch=sm_30 $< -o $@
dct8x8.o: dct8x8.cu
	@echo Compilling dct8x8.cu
	@nvcc $(addprefix -I,$(INCLUDE)) -m64 -c -arch=sm_30 $< -o $@

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executables ...
	-@rm -f $(TARGETS)
