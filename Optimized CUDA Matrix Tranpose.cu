#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>
#define TILE_DIM_N 16
#define TILE_DIM 32
#define BLK_DIM_Y 16
//Error handle
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort=true)
{
       if (code != cudaSuccess)
       {
         fprintf(stderr,"GPUassert: %s %s %d\n",
         cudaGetErrorString(code), file, line);
         if (abort) exit(code);
       }
}

// time stamp function in seconds
double getTimeStamp() {
    struct timeval  tv ;
    gettimeofday( &tv, NULL  ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}
 // host side matrix addition
void h_transposemat(int *A, int *B, int nx, int ny, int sel) {
  int idx1;
  int idx2;
  if (sel != 3){
    for (int i=0;i<nx;i++){
      for(int j=0;j<ny;j++){
        idx1=(j*nx)+i;
        idx2=(i*ny)+j;
        B[idx1]=A[idx2];
      }
    }
  }
  else{
    for (int i=0;i<nx;i++){
      for(int j=0;j<ny;j++){
        idx1=(j*nx)+i;
        idx2=(i*ny)+j;
        B[idx2]=A[idx1];
      }
    }
  }
}
 // device-side matrix addition
template <int sel>
__global__ void d_transposemat( int *A, int *B, int nx, int ny  ){

  if (sel == 1){
    int ix = threadIdx.x + blockIdx.x*blockDim.x ;
    int iy = threadIdx.y + blockIdx.y*blockDim.y ;
    if( (ix<nx) && (iy<ny)  ){
      int idx1 = (iy*nx) + ix;
      int idx2 = (ix*ny) + iy;
      B[idx1] = A[idx2];
    }
  }
  else if(sel == 2){
    __shared__ int tile[TILE_DIM][TILE_DIM+1];

  	int ix = blockIdx.x * TILE_DIM + threadIdx.x;
  	int iy = blockIdx.y * TILE_DIM + threadIdx.y;
  	int a = TILE_DIM;
  	int b = TILE_DIM;
    a = (a > (nx - iy))?(nx - iy):TILE_DIM;
  	if ( ix < ny )
  		for (int j = 0; j < a; j+= BLK_DIM_Y)
  			tile[threadIdx.y+j][threadIdx.x] = A[(iy+j)*ny + ix];
    ix = blockIdx.y * TILE_DIM + threadIdx.x;
  	iy = blockIdx.x * TILE_DIM + threadIdx.y;
  	__syncthreads();
    b = (b > (ny - iy))?(ny - iy):TILE_DIM;
  	if ( ix < nx)
  		for (int j = 0; j < b; j += BLK_DIM_Y)
  			B[(iy+j)*nx + ix] = tile[threadIdx.x][threadIdx.y + j];
  }

    else{
      __shared__ int block[TILE_DIM * (TILE_DIM + 1)];
      int xB = blockDim.x * blockIdx.x;
      int yB = blockDim.y * blockIdx.y;
      int ix = xB + threadIdx.x;
      int iy = yB + threadIdx.y;
      int oIdx,index_out;

      if(ix<nx && iy<ny){
        int index_in = (nx*iy)+ix;
        int iIdx = threadIdx.y*(TILE_DIM+1)+threadIdx.x;
        block[iIdx] = A[index_in];
      }
      __syncthreads();
      oIdx = threadIdx.x*(TILE_DIM+1)+threadIdx.y;
      index_out = ny*(xB+threadIdx.y)+yB+threadIdx.x;
      if(ix<nx && iy<ny)
        B[index_out] = block[oIdx];
    }
}

int main( int argc, char *argv[]  ) {

    // get program arguments
    if( argc != 3 ) {
        printf("Error: wrong number of args\n");
        exit(0);
    }
    int bytes, noElems;
    int sel;
    int idx;
    double timeStampC;
    double timeStampD;
    int nx = atoi( argv[1]  ) ;
    int ny = atoi( argv[2]  ) ;
    if (nx < ny) {
        nx = nx + ny;
        ny = nx - ny;
        nx = nx - ny;
    }
    noElems = nx*ny ;
    bytes = noElems * sizeof(int) ;

    //Determining sel(selection)
    if((noElems < 1600) || (noElems > 500000000))
      sel = 1;
    else if(nx == ny)
      sel = 2;
    else if((nx % 32 == 0) && (ny % 32 == 0))
      sel = 3;
    else
      sel = 1;


    // alloc memory host-side
    int *h_A, *h_dB;
    gpuErrchk(cudaMallocHost( (int **) &h_A, bytes  )) ;
    gpuErrchk(cudaMallocHost( (int **) &h_dB, bytes )) ;
    int *h_hB = (int *) malloc( bytes   ) ;

    // init matrices with random data
    for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            idx=(i*ny)+j;
            h_A[idx]=rand();
        }
    }

    //host side matrix addition
    // double time_testA = getTimeStamp();
    h_transposemat(h_A,h_hB,nx,ny,sel);
    // double time_testB = getTimeStamp();

    // alloc memory dev-side
    int *d_A, *d_B ;
    gpuErrchk(cudaMalloc( (void **) &d_A, bytes )) ;
    gpuErrchk(cudaMalloc( (void **) &d_B, bytes )) ;

    double timeStampA = getTimeStamp() ;

    //transfer data to dev
    gpuErrchk(cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice  )) ;

    double timeStampB = getTimeStamp() ;

    // invoke Kernel
    dim3 block1(TILE_DIM_N, TILE_DIM_N);
    dim3 grid1( (nx + block1.x-1)/block1.x, (ny + block1.x-1)/block1.x  ) ;
    dim3 block2(TILE_DIM, BLK_DIM_Y);
    dim3 grid2( (nx + block2.x-1)/block2.x, (ny + block2.x-1)/block2.x  ) ;
    dim3 block3(TILE_DIM, TILE_DIM);
    dim3 grid3( (nx + block3.x-1)/block3.x, (ny + block3.x-1)/block3.x  ) ;

     // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
     //         grid.x, grid.y, grid.z, block.x, block.y, block.z);
    switch (sel){
      case 1:
        timeStampC = getTimeStamp() ;
        d_transposemat<1><<<grid1, block1>>>( d_A, d_B, nx, ny  );
        timeStampD = getTimeStamp() ;
        break;
      case 2:
        timeStampC = getTimeStamp() ;
        d_transposemat<2><<<grid2, block2>>>( d_A, d_B, nx, ny  );
        timeStampD = getTimeStamp() ;
        break;
      case 3:
        timeStampC = getTimeStamp() ;
        d_transposemat<3><<<grid3, block3>>>( d_A, d_B, nx, ny  );
        timeStampD = getTimeStamp() ;
        break;
    }

    double timeStampE = getTimeStamp() ;
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()) ;

    //copy data back
    gpuErrchk(cudaMemcpy( h_dB, d_B, bytes, cudaMemcpyDeviceToHost  )) ;

    double timeStampF = getTimeStamp() ;

    // check result
    int check_flag=0;
    for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            int idx=(i*ny)+j;
            if (h_hB[idx]!=h_dB[idx]){
                check_flag=1;
            }
        }
    }
    gpuErrchk(cudaFreeHost( h_A ));
    gpuErrchk(cudaFreeHost( h_dB));
    gpuErrchk(cudaFree( d_A )) ;
    gpuErrchk(cudaFree( d_B )) ;
    gpuErrchk(cudaDeviceReset()) ;

    // print out results
    if(check_flag==1){
        printf("Error: h_hB and h_dB are not equal !\n");
        exit(0);
    }
    else{
        // double cpu_exec_time = time_testB - time_testA;
        double total_time = (timeStampB - timeStampA) +
                            (timeStampD - timeStampC) +
                            (timeStampF - timeStampE) ;
        // printf("CPU Exec-- %.6f\n",cpu_exec_time);
        printf("%.6f\n",total_time);
    }
}
