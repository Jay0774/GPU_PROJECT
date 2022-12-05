#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<bits/stdc++.h>

using namespace std;
using namespace cv;

// Note : All cout statements are purely for debugging purpose

int comparev1(Mat in, Mat out)
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

int comparev2(int *r1, int *g1, int *b1, int *r2, int *g2, int *b2, int N)
{
        // imput two image matrices in and out
        // comparing every pixel for checking that images are same or not
        int i,j;
        for(i=0;i<N;i++)
        {
                for(j=0;j<N;j++)
                {
                        if(r1[i*N+j]!=r2[i*N+j] || g1[i*N+j]!=g2[i*N+j] || b1[i*N+j]!=b2[i*N+j])
                                break;
                }
        }
        // return 1 if images are same else return 0
        if(i==N && j==N)
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


int main()
{
        // p,q = keys used for arnold cat map
        // k = key used for encryption
        int p=2,q=5,k=13,iter=30,i,j,N=510,new_i,new_j;
        int *image_r,*image_b,*image_g,*k_image_r,*k_image_b,*k_image_g,*k_result_r,*k_result_b,*k_result_g,*p1,*p2,*n;
        float ms,total=0;

        //variables for measuring time
        clock_t start,stop;


        // reading input image
        Mat original = imread("test.png");
        Mat original_resized,original_resized_gray,encrypted,decrypted;

        // resizing the image
        cout<<"Resizing Image"<<endl;
        original_resized = Resize(original,N);
        cout<<original_resized.rows<<endl;
        //cvtColor(original_resized,original_resized_gray,CV_BGR2GRAY);
        imwrite("test/Resized.png",original_resized);
        //imwrite("Resized_gray.png",original_resized_gray);
        cout<<"Resizing Image Done"<<endl;

        //print_data(original_resized);
        //test_acm();
        image_r = (int*)malloc(N*N*sizeof(int));
        image_b = (int*)malloc(N*N*sizeof(int));
        image_g = (int*)malloc(N*N*sizeof(int));

        k_image_r = (int*)malloc(N*N*sizeof(int));
        k_image_g = (int*)malloc(N*N*sizeof(int));
        k_image_b = (int*)malloc(N*N*sizeof(int));

        for(i=0;i<N;i++)
        {
                for(j=0;j<N;j++)
                {
                        k_image_r[N*i+j] = int(original_resized.at<Vec3b>(i,j)[0]);
                        k_image_g[N*i+j] = int(original_resized.at<Vec3b>(i,j)[1]);
                        k_image_b[N*i+j] = int(original_resized.at<Vec3b>(i,j)[2]);
                        image_r[N*i+j] = int(original_resized.at<Vec3b>(i,j)[0]);
                        image_g[N*i+j] = int(original_resized.at<Vec3b>(i,j)[1]);
                        image_b[N*i+j] = int(original_resized.at<Vec3b>(i,j)[2]);
                        //cout<<image_r[N*i+j]<<" ";
                }
                //cout<<endl;
        }


        // starting encryption
        start = clock();
        cout<<"Encryption Start"<<endl;
        while(iter>0)
        {
                for(i=0;i<N;i++)
                {
                        for(j=0;j<N;j++)
                        {
                                        new_i = (i+p*j)%N;
                                        new_j = (i*q+(1+p*q)*j)%N;
                                        image_r[N*i+(N-1-j)] = image_r[N*new_i+(N-1-new_j)];
                                        image_g[N*i+(N-1-j)] = image_g[N*new_i+(N-1-new_j)];
                                        image_b[N*i+(N-1-j)] = image_b[N*new_i+(N-1-new_j)];
                        }
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
          
                string s = "test/encrypted"+to_string(iter)+".png";
                imwrite(s,original_resized);
		
                iter--;
        }
        
        imwrite("test/Encrypted.png",original_resized);
        stop = clock();
        cout<<"Encryption Done"<<endl;
        cout<<"Time Taken for Encryption on CPU is "<<(double)(stop-start)/(CLOCKS_PER_SEC)*1e3<<" milliseconds"<<endl;

        // starting decryption
        cout<<"Decryption Start"<<endl;
        start = clock();
        total = 0;
        iter = 0;
        while(!comparev2(image_r,image_g,image_b,k_image_r,k_image_g,k_image_b,N))
        {
                cout<<iter<<" "<<comparev2(image_r,image_g,image_b,k_image_r,k_image_g,k_image_b,N)<<endl;
                for(i=0;i<N;i++)
                {
                        for(j=0;j<N;j++)
                        {
                                        new_i = (i+p*j)%N;
                                        new_j = (i*q+(1+p*q)*j)%N;
                                        image_r[N*i+(N-1-j)] = image_r[N*new_i+(N-1-new_j)];
                                        image_g[N*i+(N-1-j)] = image_g[N*new_i+(N-1-new_j)];
                                        image_b[N*i+(N-1-j)] = image_b[N*new_i+(N-1-new_j)];
                        }
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
		
                string s = "test/decrypted"+to_string(iter)+".png";
                imwrite(s,original_resized);
                
                iter++;
        }
        
        imwrite("test/Decrypted.png",original_resized);
        stop = clock();
        cout<<"Decryption Done"<<endl;
        cout<<"Time Taken for Decryption on CPU is "<<(double)(stop-start)/(CLOCKS_PER_SEC)*1e3<<" milliseconds"<<endl;
        return 0;
}
