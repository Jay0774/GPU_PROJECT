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

Mat Resize(Mat in,int size)
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

Mat encrypt(Mat in,int iterations,int p,int q,int key)
{
	// function for encryption
	// input input image, number of iterations, key values used for arnold cat map, key value for encryption
	// return encrypted image  
	Mat out;
	out = in.clone();
	//cout<<compare(in,imread("test1/Resized.png"))<<endl;
	int i,j,newx,newy,n=in.rows;
	while(iterations--)
	{
		//cout<<iterations<<" start"<<endl;
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{
				// calculating new indices 
				newx = (i + p*j)%n;
				newy = (i*q + (1+p*q)*j)%n;
				//out.at<Vec3b>(newx,n-newy-1) = in.at<Vec3b>(i,n-j-1);
				// performing arnold cat map
				out.at<Vec3b>(newx,n-newy-1)[0] = int(in.at<Vec3b>(i,n-j-1)[0]);
				out.at<Vec3b>(newx,n-newy-1)[1] = int(in.at<Vec3b>(i,n-j-1)[1]);
				out.at<Vec3b>(newx,n-newy-1)[2] = int(in.at<Vec3b>(i,n-j-1)[2]);
			}
		}
		//cout<<iterations<<" end"<<endl;
		//cout<<"Comparing Images "<<compare(out,in)<<endl;
		string s = "test1/e"+to_string(iterations)+".png";
		imwrite(s,out);
		in.release();
		//out.release();
		in = imread(s);
	}
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			out.at<Vec3b>(i,j)[0] = int(out.at<Vec3b>(i,j)[0])^key;
			out.at<Vec3b>(i,j)[0] = int(out.at<Vec3b>(i,j)[0])^key;
			out.at<Vec3b>(i,j)[0] = int(out.at<Vec3b>(i,j)[0])^key;
		}
	}
	return out;
}


Mat decrypt(Mat in, Mat origin,int p,int q,int key)
{
	// function for decryption
	// input encrypted images, keys for arnold map. key used for encryption
	// return decrypted image	
	int i,j,x,y,newx,newy,n=in.rows;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			in.at<Vec3b>(i,j)[0] = int(in.at<Vec3b>(i,j)[0])^key;
			in.at<Vec3b>(i,j)[0] = int(in.at<Vec3b>(i,j)[0])^key;
			in.at<Vec3b>(i,j)[0] = int(in.at<Vec3b>(i,j)[0])^key;
		}
	}
	Mat out;
	out = in.clone();
	//cout<<"Comparing encrypted and origin "<<compare(in,origin)<<endl;
	//print_data(out);
	int count = 0;
	while(!compare(out,imread("test1/Resized.png")))
	{
		//cout<<"Iteration "<<++count<<endl;
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{
				// calculating new indices
				newx = (i + p*j)%in.rows;
				newy = (i*q + (1+p*q)*j)%in.cols;
				//out.at<Vec3b>(newx,n-newy-1) = in.at<Vec3b>(i,n-j-1);
				out.at<Vec3b>(newx,n-newy-1)[0] = int(in.at<Vec3b>(i,n-j-1)[0]);
				out.at<Vec3b>(newx,n-newy-1)[1] = int(in.at<Vec3b>(i,n-j-1)[1]);
				out.at<Vec3b>(newx,n-newy-1)[2] = int(in.at<Vec3b>(i,n-j-1)[2]);
				//cout<<out.at<double>(newy,newx)<<endl;
			}
			
		}
		//cout<<compare(out,imread("test1/Resized.png"))<<endl;
		string s = "test1/d"+to_string(count)+".png";
		imwrite(s,out);
		in.release();
		//out.release();
		in = imread(s);
		//out = in.clone();
	}
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

void test_acm()
{
	// functons for testing the number of iterations required for different keys in case of arnold map 
	Mat in,out,temp;
	int count = 0,i,j,newx,newy,p=2,q=5;
	in = imread("test1/Resized.png");
	//cvtColor(in,temp,CV_BGR2GRAY);
	out = in.clone();
	//cout<<compare(in,out)<<endl;
	//cout<<compare(in,out)<<endl;
	do
	{
		out =  in.clone();
		for(i=0;i<in.rows;i++)
		{
			for(j=0;j<in.cols;j++)
			{
				newx = (i + p*j)%in.rows;
				newy = (i*q + (1+p*q)*j)%in.cols;
				//Vec3b p0 = out.at<Vec3b>(newx,in.rows-newy-1);
				//Vec3b p1 = temp.at<Vec3b>(i,in.rows-j-1);
				//p0[0] = p1[0];
				//p0[1] = p1[1];
				//p0[2] = p1[2];	
				out.at<Vec3b>(newx,in.rows-newy-1)[0] = int(in.at<Vec3b>(i,in.rows-j-1)[0]);
				out.at<Vec3b>(newx,in.rows-newy-1)[1] = int(in.at<Vec3b>(i,in.rows-j-1)[1]);
				out.at<Vec3b>(newx,in.rows-newy-1)[2] = int(in.at<Vec3b>(i,in.rows-j-1)[2]);
				//cout<<out.at<double>(newy,newx)<<endl;
			}
			
		}
		//cvtColor(out,out,CV_BGR2GRAY);
		string s = "test1/img"+to_string(++count)+".png";
		//cout<<count<<" "<<compare(out,in)<<endl;
		imwrite(s,out);
		in.release();
		//out.release();
		in = imread(s);
		count++;
	}while(!compare(out,imread("test1/Resized.png")));
	cout<<"Number of iterations required are: "<<count<<endl;
}

int main()
{
	int N=500,test_cases=0,total_time_encryption=0, total_time_decryption=0;
	for(;test_cases<=10;test_cases++)
	{
	cout<<"For N = "<<N<<endl;
	// p,q = keys used for arnold cat map
	// k = key used for encryption
	int p=2,q=5,k=13,iter=30;
	// variables for measuring time
	clock_t start,stop;

	// reading input image
	Mat original = imread("test.png");
	Mat original_resized,original_resized_gray,encrypted,decrypted;
	
	// resizing the image
	cout<<"Resizing Image"<<endl;
	original_resized = Resize(original,N);
	//cvtColor(original_resized,original_resized_gray,CV_BGR2GRAY);
	imwrite("test1/Resized.png",original_resized);
	//imwrite("Resized_gray.png",original_resized_gray);
	cout<<"Resizing Image Done"<<endl;
	
	//print_data(original_resized);
	//test_acm();
	
	// starting encryption
	cout<<"Encryption Start"<<endl;
	start = clock();
	encrypted = encrypt(original_resized,iter,p,q,k);
	stop = clock();
	imwrite("test1/Encrypted.png",encrypted);
	cout<<"Encryption Done"<<endl;
	cout<<"Time take for Encryption in CPU is: "<<(double)(stop-start)/(CLOCKS_PER_SEC)*1e3<<" milliseconds"<<endl;
	total_time_encryption += (double)(stop-start)/(CLOCKS_PER_SEC)*1e3;
	// starting decryption
	cout<<"Decryption Start"<<endl;
	start = clock();
	decrypted = decrypt(encrypted,original_resized,p,q,k);
	stop = clock();
	imwrite("test1/Decrypted.png",decrypted);
	cout<<"Decryption Done"<<endl;	
	cout<<"Time take for Decryption in CPU is: "<<(double)(stop-start)/(CLOCKS_PER_SEC)*1e3<<" milliseconds"<<endl;
	total_time_decryption += (double)(stop-start)/(CLOCKS_PER_SEC)*1e3;
	N+=50;
	}
	cout<<"Test Cases "<<test_cases<<endl;
	cout<<"Average Time for Encryption "<<(double)total_time_encryption/test_cases<<endl;
	cout<<"Average Time for Decryption "<<(double)total_time_decryption/test_cases<<endl;
	return 0;
}
