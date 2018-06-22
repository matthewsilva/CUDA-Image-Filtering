#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "tiffio.h"

// saves TIFF file from data in `raster`
void save_tiff(const char *fname, uint32 *raster, uint32 w, uint32 h) {
    TIFF *tif = TIFFOpen(fname, "w");
    if (! raster) {
        throw std::runtime_error("Could not open output file");
    }
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFWriteEncodedStrip(tif, 0, raster, w*h*4);
    TIFFClose(tif);
}

// loads image data from `fname` (allocating dynamic memory)
// *w and *h are updated with the image dimensions
// raster is a matrix flattened into an array using row-major order
// every uint32 in the array is 4 bytes, enconding 8-bit packed ABGR
// A: transparency attribute (can be ignored)
// B: blue pixel
// G: green pixel
// R: red pixel
uint32 *load_tiff(const char *fname, uint32 *w, uint32 *h) {
    TIFF *tif = TIFFOpen(fname, "r");
    if (! tif) {
        throw std::runtime_error("Could not open input file");
    }
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
    uint32 *raster = (uint32 *) _TIFFmalloc(*w * *h * sizeof (uint32));
    if (! raster) {
        TIFFClose(tif);
        throw std::runtime_error("Memory allocation error");
    }
    if (! TIFFReadRGBAImageOriented(tif, *w, *h, raster, ORIENTATION_TOPLEFT, 0)) {
        TIFFClose(tif);
        throw std::runtime_error("Could not read raster from TIFF image");
    }
    TIFFClose(tif);
    return raster;
}

// Clamp function able to be used on the GPU
__device__ void cudaClamp(float *val) {
    if (*val < 0) *val = 0;
    if (*val > 255) *val = 255;
}

void clamp(float *val) {
    if (*val < 0) *val = 0;
    if (*val > 255) *val = 255;
}

// Kernel for calculation of one pixel in the raster
__global__ void convolve(uint32 *raster, uint32 *copy,
    int w, int h, const float *filter, int st, int d) {

	// Calculate the row and column indices of the pixel to handle based on the block and thread
    int row = st + (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = st + (blockIdx.x * blockDim.x) + threadIdx.x;
    
	// Check to make sure we are in a valid part of the grid (blocks overflow off the grid)
    if (row < h-st && col < w-st) {
    
    	int index = row*w + col;
	    int startIndex = index - (st*w) - st;
    	
    	// Accumulate RGB values
    	float sumR, sumG, sumB;
        uint32 idx, pixel, kd;
        sumR = sumG = sumB = 0;
        for (uint32 k = 0 ; k < d ; k ++) {
        
            idx = startIndex + k*w;
            kd = k*d;
            for (uint32 l = 0 ; l < d ; l++) {
                pixel = copy[idx++];
                sumR += (filter[kd + l] * TIFFGetR(pixel));
                sumG += (filter[kd + l] * TIFFGetG(pixel));
                sumB += (filter[kd + l] * TIFFGetB(pixel));
            }
        }
        // Check that RGB channels to write to the raster are not <0 or >255
        cudaClamp(&sumR);
        cudaClamp(&sumG);
        cudaClamp(&sumB);
        
        // Write the ARGB channels to the pixel using bitwise shifts and ORing of bits
        raster[index] = TIFFGetA(raster[index]) << 24 | ((uint32) sumB << 16) | ((uint32) sumG << 8) | ((uint32) sumR);
    }
}

void filter_image_seq(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len) {
    // to get RGB values from a pixel, you can either use bitwise masks
    // or rely on the following macros:
    // TIFFGetR(raster[i]) red
    // TIFFGetG(raster[i]) green
    // TIFFGetB(raster[i]) blue
    // TIFFGetA(raster[i]) this value should be ignored
    //
    // to modify RGB values from a pixel, you can use bitwise shifts or masks
    // each pixel stores values in the order ABGR
    //
    // TODO: here you will filter the image in raster
    //
    uint32 *copy = new uint32[w*h];
    std::memcpy(copy, raster, sizeof(uint32)*w*h);
    uint32 d = (uint32) std::sqrt(f_len);
    uint32 idx, pixel;
    uint32 st = d / 2;
    uint32 end_w = w - d/2;
    uint32 end_h = h - d/2;
    float sumR, sumG, sumB;
    // applies filter
    for (uint32 i = st ; i < end_h ; i++) {
        for (uint32 j = st ; j < end_w ; j++) {
            sumR = sumG = sumB = 0;
            for (uint32 k = 0 ; k < d ; k ++) {
                idx = (i-st+k)*w + (j-st);
                for (uint32 l = 0 ; l < d ; l++) {
                    pixel = copy[idx++];
                    sumR += (filter[k*d + l] * TIFFGetR(pixel));
                    sumG += (filter[k*d + l] * TIFFGetG(pixel));
                    sumB += (filter[k*d + l] * TIFFGetB(pixel));
                }
            }
            clamp(&sumR);
            clamp(&sumG);
            clamp(&sumB);
            raster[i*w + j] = TIFFGetA(raster[i*w + j]) << 24 | ((uint32) sumB << 16) | ((uint32) sumG << 8) | ((uint32) sumR);
        }
    }
    delete [] copy;
}

void filter_image_par(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len, int blockSize) {
    //
    // TODO: here you will filter the image in raster using GPU threads
    //

    // to get RGB values from a pixel, you can either use bitwise masks
    // or rely on the following macros:
    // TIFFGetR(raster[i]) red
    // TIFFGetG(raster[i]) green
    // TIFFGetB(raster[i]) blue
    // TIFFGetA(raster[i]) this value should be ignored
    //
    // to modify RGB values from a pixel, you can use bitwise shifts or masks
    // each pixel stores values in the order ABGR
    //
    // TODO: here you will filter the image in raster
    //
    uint32 *copy = new uint32[w*h];
    std::memcpy(copy, raster, sizeof(uint32)*w*h);
    uint32 d = (uint32) std::sqrt(f_len);
    uint32 st = d / 2;
    uint32 end_w = w - d/2;
    uint32 end_h = h - d/2;
    
    // Declare the device versions of the arrays
    uint32 *dev_raster;
    float *dev_filter;
    uint32 *dev_copy;
    
    // Variable to check for CUDA errors
    cudaError_t status;

	// Choose GPU to run
    status = cudaSetDevice(0);
    if (status != cudaSuccess) std::cerr << "cudaSetDevice failed!" << std::endl;

	

    // Allocate space for the arrays in the GPU
    status = cudaMalloc(&dev_raster, sizeof(uint32) * (w*h));
    if (status != cudaSuccess) std::cerr << "cudaMalloc (in) failed!" << std::endl;
    status = cudaMalloc(&dev_copy, sizeof(uint32) * (w*h));
    if (status != cudaSuccess) std::cerr << "cudaMalloc (in) failed!" << std::endl;
    status = cudaMalloc(&dev_filter, sizeof(float) * f_len);
    if (status != cudaSuccess) std::cerr << "cudaMalloc (in) failed!" << std::endl;
    
    // Transfer image raster and filter data from CPU to GPU
    status = cudaMemcpy(dev_raster, raster, sizeof(float) * (w*h), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) std::cerr << "cudaMemcpy H2D failed!" << std::endl;
    status = cudaMemcpy(dev_copy, raster, sizeof(float) * (w*h), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) std::cerr << "cudaMemcpy H2D failed!" << std::endl;
    status = cudaMemcpy(dev_filter, filter, sizeof(float) * (f_len), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) std::cerr << "cudaMemcpy H2D failed!" << std::endl;

	// Defines how many threads per block there are in the x and y dimensions
    dim3 threadsPerBlock(blockSize, blockSize, 1);
    
    // Computes how many blocks will fit into the image with one thread per pixel
    // Overflows past the end of the image rows and columns when the image dimensions (x&y) don't divide evenly by the block dimensions (x&y) 
    dim3 numBlocks((int)std::ceil((float)(end_w-st)/(float)threadsPerBlock.x),
                    (int)std::ceil((float)(end_h-st)/(float)threadsPerBlock.y), 1);

    // Do the work in the GPU
    convolve<<<numBlocks, threadsPerBlock>>>(dev_raster, dev_copy, w, h, dev_filter, st, d);

	// Wait for the kernel to finish, and check for errors
    status = cudaThreadSynchronize();
    if (status != cudaSuccess) std::cerr << "error code " << status << " returned after kernel!" << std::endl;


    // Transfer resulting image raster from GPU to CPU
    status = cudaMemcpy(raster, dev_raster, sizeof(uint32) * (w*h), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) std::cerr << "cudaMemcpy D2H failed!" << std::endl;

    // Free the memory allocated in the GPU
    cudaFree(dev_raster);
    cudaFree(dev_copy);
    cudaFree(dev_filter);
    
    // Free the copy we made
    free(copy);
}

float *load_filter(const char *fname, int *n) {
    std::ifstream myfile(fname);
    if (! myfile) {
        throw std::runtime_error("Could not open filter file");
    }
    myfile >> *n;
    float *filter = new float[*n];
    for (int i = 0 ; i < *n ; i++) myfile >> filter[i];
    myfile.close();
    return filter;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage:\t./filter <in_fname> <out_fname> <filter_fname> <algo>" << std::endl;
        std::cout << "<in_fname> path to the input image" << std::endl;
        std::cout << "<out_fname> path to the output image" << std::endl;
        std::cout << "<filter_fname> path to the filter file" << std::endl;
        std::cout << "<algo> whether to use the sequential (seq) or parallel algorithm (par)" << std::endl;
        return 0;
    }

	// Defines the x&y dimension of the thread block used in the parallel solution
	int blockSize;
	sscanf(argv[5], "%d", &blockSize);

    uint32 width, height;

    // loads the filter
    int f_len;
    float *filter = load_filter(argv[3], &f_len);

    // loads image bytes from file name supplied as a command line argument
    // this function allocates memory dynamically
    uint32 *image = load_tiff(argv[1], &width, &height);

	// Make a malloc in the GPU to load the CUDA library and make sure that it is working properly
	uint32 *dev_initCuda;
    cudaError_t status;
    status = cudaMalloc(&dev_initCuda, 1);
    if (status != cudaSuccess) std::cerr << "cudaMalloc (in) failed!" << std::endl;

    // measure time of the algorithm
    auto start = std::chrono::high_resolution_clock::now();
    if (! std::strcmp(argv[4], "seq")) {
        // call the sequential implementation
        filter_image_seq(image, width, height, filter, f_len);
    } else if (! std::strcmp(argv[4], "par")) {
        // TODO: call the parallel implementation
        filter_image_par(image, width, height, filter, f_len, blockSize);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count();

    // save new file with filtered image
    save_tiff(argv[2], image, width, height);

    // frees memory allocated by load_filter and load_tiff
    delete [] filter;
    _TIFFfree(image);

    return 0;
}
