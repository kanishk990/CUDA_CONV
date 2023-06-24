
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <ctime>
#include <cstdlib>
#include <cmath>

#define maskCols 3
#define maskRows 3
#define imgchannels 1
#define TILE_SIZE 32
#define SM (TILE_SIZE + maskRows-1)

using namespace std;

void sequentialConvolution(const unsigned char*inputImage,const float * kernel ,unsigned char * outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;                         // center index of kernel
    float sum;                                      // accumulation variable
    int rowIndex, colIndex;                         // indice di riga e di colonna

    const unsigned char * inputImageData = inputImage;
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    // cout << kCenterX << " " << kCenterY << endl;

    for (int k=0; k<channels; k++) {                    //cycle on channels
        for (i = 0; i < dataSizeY; ++i)                //cycle on image rows
        {
            for (j = 0; j < dataSizeX; ++j)            //cycle on image columns
            {
                sum = 0;
                for (m = 0; m < kernelSizeY; ++m)      //cycle kernel rows
                {
                    mm = kernelSizeY - 1 - m;       // row index of flipped kernel

                    for (n = 0; n < kernelSizeX; ++n)  //cycle on kernel columns
                    {
                        nn = kernelSizeX - 1 - n;   // column index of flipped kernel

                        // indexes used for checking boundary
                        rowIndex = i + m - kCenterY;
                        colIndex = j + n - kCenterX;

                        // ignore pixels which are out of bound
                        if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                            sum += inputImageData[(dataSizeX * rowIndex + colIndex)*channels + k] * kernel[kernelSizeX * mm + nn];
                    }
                }
                outputImageData[(dataSizeX * i + j)*channels + k] = sum;

            }
        }
    }
}
__global__ void convKernel(unsigned char * inputImage, const float * __restrict__ kernel, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels){

    int m, n;
    float sum;
    // int rowIndex, colIndex;
    int kCenterX, kCenterY; 

    const unsigned char * inputImageData = inputImage;

    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    __shared__ unsigned char s_ImageData[SM][SM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int temp = TILE_SIZE * ty + tx;
    int row = temp/SM;
    int col = temp%SM;
    int tempX = bx*TILE_SIZE + col - kCenterX;
    int tempY = by*TILE_SIZE + row - kCenterY;
    
    if ( tempY < dataSizeY && tempX >= 0 && tempY >= 0 && tempX < dataSizeX) {
        s_ImageData[row][col] = inputImageData[tempY * dataSizeY + tempX];
    }
    else {
        s_ImageData[row][col] = 0;
    }

    temp = TILE_SIZE * ty + tx + TILE_SIZE * TILE_SIZE;
    row = temp/SM;
    col = temp%SM;
    tempX = bx*TILE_SIZE + col - kCenterX;
    tempY = by*TILE_SIZE + row - kCenterY;
    
    if ( tempX < dataSizeX && tempY < dataSizeY && tempX >= 0 && tempY >= 0 && row < SM ) {
        s_ImageData[row][col] = inputImageData[tempY * dataSizeY + tempX];
    }
    else if ( row < SM ) {
        s_ImageData[row][col] = 0;
    }

    // int row = by * blockDim.y + ty;
    // int col = bx * blockDim.x + tx;

    // s_ImageData[ty][tx] = inputImageData[col * dataSizeY + row];

    __syncthreads();

    sum = 0;
    for (m = 0; m < kernelSizeY; ++m)      //cycle kernel rows
    {
    //     mm = kernelSizeY - 1 - m;       // row index of flipped kernel

        for (n = 0; n < kernelSizeX; ++n)  //cycle on kernel columns
        {
    //         nn = kernelSizeX - 1 - n;   // column index of flipped kernel

    //         // indexes used for checking boundary
    //         rowIndex = tx + m - kCenterY;
    //         colIndex = ty + n - kCenterX;

    //         // ignore pixels which are out of bound
    //         if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
    //             // sum += inputImageData[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
                sum += s_ImageData[ty+m][tx+n] * kernel[kernelSizeX * m + n];
        }
    }
    outputImageData[dataSizeX * (by * TILE_SIZE + ty) + (bx * TILE_SIZE + tx)] = sum;
}

__global__ void SobelEdgeDetector(unsigned char * inputImage, const float * __restrict__ kernel, const float * __restrict__ kernel2, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels){

    int m, n;
    float sum, sum2;
    // int rowIndex, colIndex;
    int kCenterX, kCenterY; 

    const unsigned char * inputImageData = inputImage;

    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    __shared__ unsigned char s_ImageData[SM][SM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int temp = TILE_SIZE * ty + tx;
    int row = temp/SM;
    int col = temp%SM;
    int tempX = bx*TILE_SIZE + col - kCenterX;
    int tempY = by*TILE_SIZE + row - kCenterY;
    
    if ( tempX < dataSizeX && tempY < dataSizeY && tempX >= 0 && tempY >= 0 ) {
        s_ImageData[row][col] = inputImageData[tempY * dataSizeY + tempX];
    }
    else {
        s_ImageData[row][col] = 0;
    }

    temp = TILE_SIZE * ty + tx + TILE_SIZE * TILE_SIZE;
    row = temp/SM;
    col = temp%SM;
    tempX = bx*TILE_SIZE + col - kCenterX;
    tempY = by*TILE_SIZE + row - kCenterY;
    
    if ( tempX < dataSizeX && tempY < dataSizeY && tempX >= 0 && tempY >= 0 && row < SM ) {
        s_ImageData[row][col] = inputImageData[tempY * dataSizeY + tempX];
    }
    else if ( row < SM ) {
        s_ImageData[row][col] = 0;
    }

    // int row = by * blockDim.y + ty;
    // int col = bx * blockDim.x + tx;

    // s_ImageData[ty][tx] = inputImageData[col * dataSizeY + row];

    __syncthreads();

    sum = 0;
    sum2 = 0;
    for (m = 0; m < kernelSizeY; ++m)      //cycle kernel rows
    {
    //     mm = kernelSizeY - 1 - m;       // row index of flipped kernel

        for (n = 0; n < kernelSizeX; ++n)  //cycle on kernel columns
        {
    //         nn = kernelSizeX - 1 - n;   // column index of flipped kernel

    //         // indexes used for checking boundary
    //         rowIndex = tx + m - kCenterY;
    //         colIndex = ty + n - kCenterX;

    //         // ignore pixels which are out of bound
    //         if (rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
    //             // sum += inputImageData[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
            if ( kernelSizeX * m + n == 0 || kernelSizeX * m + n == 1 || kernelSizeX * m + n == 2 ) {
                sum += s_ImageData[ty+m][tx+n] * kernel[kernelSizeX * m + n];
            }
            if ( kernelSizeX * m + n == 6 || kernelSizeX * m + n == 7 || kernelSizeX * m + n == 8 ) {
                sum += s_ImageData[ty+m][tx+n] * kernel[kernelSizeX * m + n];
            }
            if ( kernelSizeX * m + n == 2 || kernelSizeX * m + n == 5 || kernelSizeX * m + n == 8 ) {
                sum2 += s_ImageData[ty+m][tx+n] * kernel2[kernelSizeX * m + n];
            }
            if ( kernelSizeX * m + n == 0 || kernelSizeX * m + n == 3 || kernelSizeX * m + n == 6 ) {
                sum2 += s_ImageData[ty+m][tx+n] * kernel2[kernelSizeX * m + n];
            }
        }
    }
    outputImageData[dataSizeX * (by * TILE_SIZE + ty) + (bx * TILE_SIZE + tx)] = ceil(sqrt((sum*sum) + (sum2*sum2)));
}

int main(){
    int width, height, bpp;
    unsigned char *image, *seq_img;

    char Images[6][15] = { "image64.png", "image128.png", "image256.png", "image512.png", "image1024.png", "image2048.png" };

    int j = 4;
    // for (j=0; j<; j++) {

        image = stbi_load( Images[j], &width, &height, &bpp, imgchannels );
        seq_img = (unsigned char*)malloc(width*height*sizeof(unsigned char));

        // cout << "Height x Width " << height << "x" << width << endl; 
    
        // float hostMaskData[maskRows*maskCols];
        // for(int i=0; i< maskCols*maskCols; i++){
        //     hostMaskData[i] = 1.0/(maskRows*maskCols);
        // }

        float hostMaskData[maskRows*maskCols];
        hostMaskData[0] = 1;
        hostMaskData[1] = 2;
        hostMaskData[2] = 1;
        hostMaskData[3] = 0;
        hostMaskData[4] = 0;
        hostMaskData[5] = 0;
        hostMaskData[6] = -1;
        hostMaskData[7] = 2;
        hostMaskData[8] = -1;

        float hostMaskData2[maskRows*maskCols];
        hostMaskData2[0] = -1;
        hostMaskData2[1] = 0;
        hostMaskData2[2] = 1;
        hostMaskData2[3] = -2;
        hostMaskData2[4] = 0;
        hostMaskData2[5] = 2;
        hostMaskData2[6] = -1;
        hostMaskData2[7] = 0;
        hostMaskData2[8] = 1;


        clock_t begin = clock();
        sequentialConvolution(image, hostMaskData, seq_img, maskRows, maskCols, width, height, imgchannels);
        clock_t end = clock();
        double elapsed_time = double(end-begin) / CLOCKS_PER_SEC;
        elapsed_time *= 1000;
        // cout << j << " CPU: " << elapsed_time << endl;
        // stbi_write_png("mynew_seq.png", width, height, imgchannels, seq_img, 0);

        // cuda Program

        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        const dim3 block_size(TILE_SIZE, TILE_SIZE);
        const dim3 num_blocks(width/TILE_SIZE, height/TILE_SIZE);

        // cout << "Block Size " << block_size.x << "x" << block_size.y << endl;
        // cout << "Num Block " << num_blocks.x << "x" << num_blocks.y << endl;

        unsigned char *d_image = 0, *d_seqimg = 0;
        float *d_hostmaskdata = 0;
        float *d_hostmaskdata2 = 0;

        cudaMalloc((void**)&d_image, sizeof(unsigned char) * width * height);
        cudaMalloc((void**)&d_seqimg, sizeof(unsigned char) * width * height);
        cudaMalloc((void**)&d_hostmaskdata, sizeof(float) * maskCols * maskRows);
        cudaMalloc((void**)&d_hostmaskdata2, sizeof(float) * maskCols * maskRows);    

        cudaMemcpy(d_image, image, sizeof(char) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hostmaskdata, hostMaskData, sizeof(float) * maskCols * maskRows, cudaMemcpyHostToDevice);
        cudaMemcpy(d_hostmaskdata2, hostMaskData2, sizeof(float) * maskCols * maskRows, cudaMemcpyHostToDevice);

        cudaEventRecord(start_kernel);
        SobelEdgeDetector<<<num_blocks, block_size>>>(d_image, d_hostmaskdata, d_hostmaskdata2, d_seqimg, maskRows, maskCols, width, height, imgchannels);
        // convKernel<<<num_blocks, block_size>>>(d_image, d_hostmaskdata, d_seqimg, maskRows, maskCols, width, height, imgchannels);
        cudaEventRecord(stop_kernel);

        cudaMemcpy(seq_img, d_seqimg, sizeof(char) * width * height, cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop_kernel);
        float k_time ;
        cudaEventElapsedTime(&k_time, start_kernel, stop_kernel);
        // cout << j << " GPU: " << k_time << endl;

        stbi_write_png("mynew_seq.png", width, height, imgchannels, seq_img, 0);    

        cout << elapsed_time/k_time << ", ";

        cudaFree(d_image);
        cudaFree(d_seqimg);
        cudaFree(d_hostmaskdata);

        free(image);
        free(seq_img);

    // }
    cout << endl;

    return 0;
}
