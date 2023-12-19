
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void NmDistanceKernel1(int b,int n,const float * xyz,const float* L, int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				float a = L[(i * n + j) * 3 * 3 + 0]; // L[i, j, 0, 0]
                float b = L[(i * n + j) * 3 * 3 + 3]; // L[i, j, 1, 0]
                float c = L[(i * n + j) * 3 * 3 + 4]; // L[i, j, 1, 1]
                float d = L[(i * n + j) * 3 * 3 + 6]; // L[i, j, 2, 0]
                float e = L[(i * n + j) * 3 * 3 + 7]; // L[i, j, 2, 1]
                float f = L[(i * n + j) * 3 * 3 + 8]; // L[i, j, 2, 2]
				int best_i=0;
				float best=0;
				#pragma unroll
				for (int k=0;k<end_k;k++){
						float x2=buf[k*3+0]-x1;
						float y2=buf[k*3+1]-y1;
						float z2=buf[k*3+2]-z1;
						float d = pow(a * x2, 2) + pow(b * x2 + c * y2, 2) + pow(d * x2 + e * y2 + f * z2, 2);
						if (k==0 || d<best){
							best=d;
							best_i=k+k2;
						}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int k2=0;k2<m;k2+=batch){
		int end_k=min(m,k2+batch)-k2;
		for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
			buf[j]=xyz2[(k2)*3+j];	
		}
		__syncthreads();
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz[(j)*3+0];
			float y1=xyz[(j)*3+1];
			float z1=xyz[(j)*3+2];
			int best_i=0;
			float best=0;
			int end_ka=end_k-(end_k&3);
			#pragma unroll
			for (int k=0;k<end_ka;k++){
					float x2=buf[k*3+0]-x1;
					float y2=buf[k*3+1]-y1;
					float z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
			}
			if (k2==0 || result[(j)]>best){
				result[(j)]=best;
				result_i[(j)]=best_i;
			}
		}
		__syncthreads();
	}
}
// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, n, xyz1.data<float>(), m, xyz2.data<float>(), dist1.data<float>(), idx1.data<int>());
	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, m, xyz2.data<float>(), n, xyz1.data<float>(), dist2.data<float>(), idx2.data<int>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;


}