#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<bits/stdc++.h>

using namespace std;
using namespace cv;

// Note : All cout statements are purely for debugging purpose

int compare(Mat in, Mat out)
{
        // imput two image matrices in and out
        // comparing every pixel for checking that images are same or not
        int i,j;
        for(i=0;i<in.rows;i++)
        {
                for(j=0;j<in.cols;j++)
                {
                        if(int(in.at<Vec3b>(i,j)[0])!=int(out.at<Vec3b>(i,j)[0]) &&
                        int(in.at<Vec3b>(i,j)[1])!=int(out.at<Vec3b>(i,j)[1]) &&
                        int(in.at<Vec3b>(i,j)[2])!=int(out.at<Vec3b>(i,j)[2]))
                                break;
                }
        }
        // return 1 if images are same else return 0
        if(i==in.rows && j==in.cols)
                return 1;
        else
                return 0;
}


Mat Resize(Mat in, int size)
{
        // input a image matrice
        // resizing the image based min(width, height)
        cout<<"Input image dimensions "<<in.rows<<" "<<in.cols<<endl;
        Mat out;
        int m;
        if(in.rows>in.cols)
                m = in.cols;
        else
                m = in.rows;
        resize(in,out,Size(size,size),0,0,INTER_LINEAR);
        cout<<"Resized image dimensions "<<out.rows<<" "<<out.cols<<endl;
        // return resized image
        return out;
}

void print_data(Mat in)
{
        // function for printing the data inside image like pixel values
        // input image
        int i,j;
        for(i=0;i<in.rows;i++)
        {
                for(j=0;j<in.cols;j++)
                {
                        cout<<"For row and column "<<i<<" "<<j<<"respectively, pixels are : "<<
                                int(in.at<Vec3b>(i,j)[0])<<" "<<
                                int(in.at<Vec3b>(i,j)[1])<<" "<<
                                int(in.at<Vec3b>(i,j)[2])<<endl;
                }
        }
}

__global__ void arnold_cat_map(int *image_r,int *image_g,int *image_b,int *k_image_r,int *k_image_g,int *k_image_b,int p,int q,int n)
{
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
       	int x,y,new_x,new_y;
	x = tid/n;
	y = tid%n;
	new_x = (x+p*y)%n;
	new_y = (x*q+(1+p*q)*y)%n;
	k_image_r[new_x*n + (n-1-new_y)] = image_r[x*n + (n-1-y)];
	k_image_g[new_x*n + (n-1-new_y)] = image_g[x*n + (n-1-y)];
	k_image_b[new_x*n + (n-1-new_y)] = image_b[x*n + (n-1-y)];
		
}

int main()
{
	// p,q = keys used for arnold cat map
	// k = key used for encryption
	int p=2,q=5,k=13,iter=30,i,j,N=600;
	int *image_r,*image_b,*image_g,*k_image_r,*k_image_b,*k_image_g,*k_result_r,*k_result_b,*k_result_g,*p1,*p2,*n;
	float ms,total=0;

	// variables for measuring time
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// reading input image
	Mat original = imread("test.png");
	Mat original_resized,original_resized_gray,encrypted,decrypted;
	
	// resizing the image
	cout<<"Resizing Image"<<endl;
	original_resized = Resize(original,N);
	//cvtColor(original_resized,original_resized_gray,CV_BGR2GRAY);
	imwrite("test/Resized.png",original_resized);
	//imwrite("Resized_gray.png",original_resized_gray);
	cout<<"Resizing Image Done"<<endl;
	
	//print_data(original_resized);
	//test_acm();
	N = original_resized.rows;
	image_r = (int*)malloc(N*N*sizeof(int));
	image_b = (int*)malloc(N*N*sizeof(int));
	image_g = (int*)malloc(N*N*sizeof(int));
	
	cudaMalloc(&k_image_r,sizeof(int)*N*N);
	cudaMalloc(&k_image_b,sizeof(int)*N*N);
	cudaMalloc(&k_image_g,sizeof(int)*N*N);
	cudaMalloc(&k_result_r,sizeof(int)*N*N);
	cudaMalloc(&k_result_b,sizeof(int)*N*N);
	cudaMalloc(&k_result_g,sizeof(int)*N*N);
	cudaMalloc(&p1,sizeof(int));
	cudaMalloc(&p2,sizeof(int));
	cudaMalloc(&n,sizeof(int));

	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			image_r[N*i+j] = int(original_resized.at<Vec3b>(i,j)[0]);
			image_g[N*i+j] = int(original_resized.at<Vec3b>(i,j)[1]);
			image_b[N*i+j] = int(original_resized.at<Vec3b>(i,j)[2]);
			//cout<<image_r[N*i+j]<<" ";
		}
		//cout<<endl;
	}


	// starting encryption
	cout<<"Encryption Start"<<endl;
	while(iter>0)
	{
		cudaEventRecord(start);
		//cudaMemcpy(p1,p,sizeof(int),cudaMemcpyHostToDevice);
		//cudaMemcpy(p2,q,sizeof(int),cudaMemcpyHostToDevice);
		//cudaMemcpy(n,N,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_r,image_r,N*N*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_g,image_g,N*N*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_b,image_b,N*N*sizeof(int),cudaMemcpyHostToDevice);
		//cudaEventRecord(start);
		arnold_cat_map<<<N,N>>>(k_image_r,k_image_g,k_image_b,k_result_r,k_result_g,k_result_b,p,q,N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		//cudaDeviceSynchronize();
		cudaMemcpy(image_r,k_result_r,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(image_g,k_result_g,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(image_b,k_result_b,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		for(i=0;i<N;i++)
		{	
			for(j=0;j<N;j++)
			{
				original_resized.at<Vec3b>(i,j)[0] = int(image_r[N*i+j]);
				original_resized.at<Vec3b>(i,j)[1] = int(image_g[N*i+j]);
				original_resized.at<Vec3b>(i,j)[2] = int(image_b[N*i+j]);
				//cout<<image_r[N*i+j]<<" ";
			}
			//cout<<endl;
		}
		string s = "test/encrypted"+to_string(iter)+".png";	
		imwrite(s,original_resized);
		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms,start,stop);
		total+=ms;
		iter--;
	}
	/*
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
				original_resized.at<Vec3b>(i,j)[0] = int(image_r[N*i+j]);
				original_resized.at<Vec3b>(i,j)[1] = int(image_g[N*i+j]);
				original_resized.at<Vec3b>(i,j)[2] = int(image_b[N*i+j]);
				//cout<<image_r[N*i+j]<<" ";
		}
		//cout<<endl;
	}
	*/
	encrypted = original_resized.clone();
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			encrypted.at<Vec3b>(i,j)[0] = int(original_resized.at<Vec3b>(i,j)[0]) ^ k;
			encrypted.at<Vec3b>(i,j)[1] = int(original_resized.at<Vec3b>(i,j)[1]) ^ k;
			encrypted.at<Vec3b>(i,j)[2] = int(original_resized.at<Vec3b>(i,j)[2]) ^ k;
		}
	}
	imwrite("test/Encrypted.png",encrypted);
	cout<<"Encryption Done"<<endl;
	cout<<"Time Taken for Encryption on GPU Kernel is "<<total<<endl;
	
	// starting decryption
	cout<<"Decryption Start"<<endl;
	total = 0;
	iter = 0;
	while(!compare(original_resized,imread("test/Resized.png")))
	{
		cudaEventRecord(start);
		//cudaMemcpy(p1,p,sizeof(int),cudaMemcpyHostToDevice);
		//cudaMemcpy(p2,q,sizeof(int),cudaMemcpyHostToDevice);
		//cudaMemcpy(n,N,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_r,image_r,N*N*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_g,image_g,N*N*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(k_image_b,image_b,N*N*sizeof(int),cudaMemcpyHostToDevice);
		//cudaEventRecord(start);
		arnold_cat_map<<<N,N>>>(k_image_r,k_image_g,k_image_b,k_result_r,k_result_g,k_result_b,p,q,N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaMemcpy(image_r,k_result_r,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(image_g,k_result_g,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(image_b,k_result_b,N*N*sizeof(int),cudaMemcpyDeviceToHost);
		for(i=0;i<N;i++)
		{	
			for(j=0;j<N;j++)
			{
				original_resized.at<Vec3b>(i,j)[0] = int(image_r[N*i+j]);
				original_resized.at<Vec3b>(i,j)[1] = int(image_g[N*i+j]);
				original_resized.at<Vec3b>(i,j)[2] = int(image_b[N*i+j]);
				//cout<<image_r[N*i+j]<<" ";
			}
			//cout<<endl;
		}
		string s = "test/decrypted"+to_string(iter)+".png";	
		imwrite(s,original_resized);
		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms,start,stop);
		total+=ms;
		iter++;
	}
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{

			original_resized.at<Vec3b>(i,j)[0] = int(image_r[N*i+j]);
			original_resized.at<Vec3b>(i,j)[1] = int(image_g[N*i+j]);
			original_resized.at<Vec3b>(i,j)[2] = int(image_b[N*i+j]);
			//cout<<image_r[N*i+j]<<" ";
		}
		//cout<<endl;
	}
	imwrite("test/Decrypted.png",original_resized);
	cout<<"Decryption Done"<<endl;
	cout<<"Time Taken for Encryption on GPU Kernel is "<<total<<endl;
	
	return 0;
}
