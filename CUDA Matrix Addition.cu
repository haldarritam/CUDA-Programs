#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>

// time stamp function in seconds 
double getTimeStamp() {     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL  ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;  
} 
 // host side matrix addition 
void h_addmat(float *A, float *B, float *C, int nx, int ny) {
    int idx;
    for (int i=0;i<nx;i++){
        for(int j=0;j<ny;j++){
            idx=(i*ny)+j;
            C[idx]=A[idx]+B[idx];            
        }
    }
}
 // device-side matrix addition 
__global__ void f_addmat( float *A, float *B, float *C, int nx, int ny  ){            
    int ix = threadIdx.x + blockIdx.x*blockDim.x ;     
    int iy = threadIdx.y + blockIdx.y*blockDim.y ;
    if( (ix<nx) && (iy<ny)  ){
        int idx = (iy*nx) + ix;
        C[idx] = A[idx] + B[idx] ;       
    }
}
int main( int argc, char *argv[]  ) {     
    // get program arguments     
    if( argc != 3 ) {
        printf("Error: wrong number of args\n");         
        exit(0);      
    }
    int bytes, noElems;
    int idx;     
    int nx = atoi( argv[1]  ) ;     
    int ny = atoi( argv[2]  ) ;
    if (nx < ny) {
        nx = nx + ny;
        ny = nx - ny;
        nx = nx - ny;
    }
    noElems = nx*ny ;
    bytes = noElems * sizeof(float) ;  
    // alloc memory host-side    
    float *h_A, *h_B, *h_dC;
    cudaMallocHost( (float **) &h_A, bytes ) ;
    cudaMallocHost( (float **) &h_B, bytes  );
    float *h_hC = (float *) malloc( bytes   ) ;
    cudaMallocHost( (float **) &h_dC, bytes ) ;           
    // init matrices with random data     
    for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            idx=(i*ny)+j;
            h_A[idx]=(i+j)/3;
            h_B[idx]=3.14*(i+j);
        }
    }
    //host side matrix addition
    // double time_testA = getTimeStamp();
    h_addmat(h_A,h_B,h_hC,nx,ny);
    // double time_testB = getTimeStamp();
    // alloc memory dev-side     
    float *d_A, *d_B, *d_C ;     
    cudaMalloc( (void **) &d_A, bytes ) ;     
    cudaMalloc( (void **) &d_B, bytes ) ;     
    cudaMalloc( (void **) &d_C, bytes ) ;
    double timeStampA = getTimeStamp() ; 
    //transfer data to dev     
    cudaMemcpy( d_A, h_A, bytes, cudaMemcpyHostToDevice  ) ;     
    cudaMemcpy( d_B, h_B, bytes, cudaMemcpyHostToDevice  ) ;    
    double timeStampB = getTimeStamp() ;
    // invoke Kernel
    dim3 block( 16, 16 ) ; 
    dim3 grid( (nx + block.x-1)/block.x, (ny + block.y-1)/block.y  ) ;
    // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    //         grid.x, grid.y, grid.z, block.x, block.y, block.z);
    f_addmat<<<grid, block>>>( d_A, d_B, d_C, nx, ny  ) ;
    cudaDeviceSynchronize() ;        
    double timeStampC = getTimeStamp() ; 
    //copy data back    
    cudaMemcpy( h_dC, d_C, bytes, cudaMemcpyDeviceToHost  ) ;
    double timeStampD = getTimeStamp() ;           
    // check result
    int check_flag=0;
    for (int i=0;i<nx;i++){
        for (int j=0;j<ny;j++){
            int idx=(i*ny)+j;           
            if (h_hC[idx]!=h_dC[idx]){
                check_flag=1;
                // printf("---  %i %i %i %f %f\n",i,j,idx,h_hC[idx],h_dC[idx]);
            }
        }
    }
    cudaFreeHost( h_A );
    cudaFreeHost( h_B );
    cudaFreeHost( h_dC);
    cudaFree( d_A   ) ; 
    cudaFree( d_B   ) ; 
    cudaFree( d_C   ) ;
    cudaDeviceReset() ;
    // print out results 
    if(check_flag==1){
        printf("Error: d_C and h_dC are not equal !\n");
        exit(0);
    }
    else{
        // double cpu_exec_time = time_testB - time_testA;
        double total_time = timeStampD - timeStampA;
        double cpu_gpu_time = timeStampB - timeStampA;
        double kernel_time = timeStampC - timeStampB;
        double gpu_cpu_time = timeStampD - timeStampC;
        // printf("CPU Exec-- %lf\n",cpu_exec_time);
        printf("%lf %lf %lf %lf\n",total_time,cpu_gpu_time,kernel_time,gpu_cpu_time);
    }
}
