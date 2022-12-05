#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<bits/stdc++.h>

using namespace std;
using namespace cv;

int main()
{
        Mat o,a,b,c,d,x;
	double max_score;
	o = imread("test.png");
	cout<<o.type()<<" "<<(o.type() & CV_MAT_DEPTH_MASK)<<endl;
        //a = imread("test/img61.jpeg");
	cout<<o.rows<<endl;
        cvtColor(o,b,CV_BGR2GRAY);
	imwrite("gray_test.png",b);
        //c = imread("test/img60.jpeg");
        //cvtColor(c,d,CV_BGR2GRAY);
        //imwrite("graya.jpeg",b);
        //imwrite("grayb.jpeg",d);
        int i,j,m=0,n=0,p=0;
        for(i=0;i<o.rows;i++)
        {
                for(j=0;j<o.cols;j++)
                {
			Vec3b p0 = o.at<Vec3b>(i,j);
			Vec3b p1 = b.at<Vec3b>(i,j);
			cout<<int(p0.val[0])<<" "<<int(p0.val[1])<<" "<<int(p0.val[2])<<endl;
			//cout<<p1.val[0]<<" "<<p1.val[1]<<" "<<p1.val[2]<<endl;
			//cout<<b.at<uchar>(i,j)<<endl;
                        //if(b.at<uchar>(i,j)!=d.at<uchar>(i,j))
                        //        break;
			if(p0[0] == p1[0])
				m++;
			if(p0[1] == p1[1])
				n++;
			if(p0[2] == p1[2])
				p++;
                }
        }
        if(i==a.rows && j==a.cols)
                cout<<"True"<<endl;
        else
                cout<<"False"<<endl;
        absdiff(b,d,x);
	cout<<m<<" "<<n<<" "<<p<<" "<<(float)(m+n+p)/(3*a.rows*a.cols)*100<<endl;
  return 0;
}
