//c++
#include <string>    
#include <vector>
#include <set> 
#include <fstream>  
#include <iostream>
using namespace std;

//opencv
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
using namespace cv;

//openmp
#include <omp.h>

struct Car_Status
{
	double time_;
	double x_;
	double y_;
	double speed_;
	bool category_;
};
//data 
struct Trace_Data
{
	std::string car_id_;
	Car_Status car_status_;
};
//read data from file
bool read_Data_From_File(const std::string &trace_file_path,std::vector<Trace_Data> &trace_data)
{
	ifstream trace_file;
	trace_file.open(trace_file_path);
	if(!trace_file.is_open())
	{
		std::cerr<<"Fail to open"<<std::endl;
		return false;
	}

	//get the first line
	std::string first_line;
	getline(trace_file,first_line);
	
	//read data
	while(!trace_file.eof())
	{
		Trace_Data trace_data_temp;
		
		trace_file>>trace_data_temp.car_id_>>trace_data_temp.car_status_.time_>>
					trace_data_temp.car_status_.x_>>trace_data_temp.car_status_.y_>>
					trace_data_temp.car_status_.speed_>>trace_data_temp.car_status_.category_;
		//std::cout<<trace_data_temp.car_id_<<std::endl;

		if(trace_data_temp.car_id_!="")
		{
			trace_data.push_back(trace_data_temp);
		}
	}

	trace_file.close();

	return true;
}


struct Car_Data
{
	std::string car_id_;
	std::vector<int> car_statuses_id_;
	double active_time_start_;
	double active_time_end_;
};
//divide the color by car_id_
bool get_Each_Car_Data(const std::vector<Trace_Data> &trace_data,std::vector<Car_Data> &car_data)
{
	//get all the car id
	std::set<std::string> car_id_set;
	for(int i=0;i<trace_data.size();i++)
	{
		car_id_set.insert(trace_data[i].car_id_);
	}

	//std::cout<<car_id_set.size()<<std::endl;
	// for(std::set<std::string>::iterator iter = car_id_set.begin();iter!=car_id_set.end();iter++)
	// {
	// 	std::cout<<*iter<<std::endl;
	// }

	//trans set into vec for openmp
	std::vector<std::string> car_id_vec;
	for(std::set<std::string>::iterator iter = car_id_set.begin();iter!=car_id_set.end();iter++)
	{
		car_id_vec.push_back(*iter);
	}
	
	// for(int i=0;i<car_id_vec.size();i++)
	// {
	// 	std::cout<<car_id_vec[i]<<std::endl;
	// }

#pragma omp parallel for
	for(int i=0;i<car_id_vec.size();i++)
	{
		Car_Data car_data_temp;
		car_data_temp.car_id_ = car_id_vec[i];

		//find all the same car_id_ in trace_data
		for(int j=0;j<trace_data.size();j++)
		{
			if(trace_data[j].car_id_==car_data_temp.car_id_ )
			{
				car_data_temp.car_statuses_id_.push_back(j);
			}
		}
#pragma omp critical 
		{
			car_data.push_back(car_data_temp);
		}
	}

	// //show for test
	// std::cout<<car_data[0].car_id_<<std::endl;
	// for(int i=0;i<car_data[0].car_statuses_.size();i++)
	// {
	// 	std::cout<<car_data[0].car_statuses_[i].time_<<std::endl;
	// }

	return true;
	
}

cv::Scalar get_Color_By_Speed(double speed,double min_speed=0,double max_speed=50)
{
	double alpha = (speed-min_speed)/(max_speed-min_speed);

	if(alpha<=0.5)
	{
		int blue = int( 255*(1-alpha*2));
		return cv::Scalar(blue,0,0);
	}
	else
	{
		int red  = int( 255*(alpha-0.5)*2    );
		return cv::Scalar(0,0,red);
	}
}

//draw one car
bool draw_Car_Trace(const std::vector<Trace_Data> &trace_data, std::vector<Car_Data> &car_data,int car_id)
{
	double x_min,x_max,y_min,y_max;
	double speed_min,speed_max;
	std::vector<cv::Point2d> points;
	std::vector<double> speeds;
	for(int i=0;i<car_data[car_id].car_statuses_id_.size();i++)
	{

		cv::Point2d point_temp;
		point_temp.x = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.x_;
		point_temp.y = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.y_;
		speeds.push_back(trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.speed_);

		//std::cout<<point_temp.x<<"\t"<<point_temp.y<<std::endl;

		points.push_back(point_temp);

		if(i==0)
		{
			x_min = point_temp.x;
			x_max = point_temp.x;
			y_min = point_temp.y;
			y_max = point_temp.y;
			speed_min = speeds[i];
			speed_max = speeds[i];

			continue;
		}
		x_min = point_temp.x< x_min ? point_temp.x : x_min;
		x_max = point_temp.x>=x_max ? point_temp.x : x_max;
		y_min = point_temp.y< y_min ? point_temp.y : y_min;
		y_max = point_temp.y>=y_max ? point_temp.y : y_max;
		speed_min = speeds[i]< speed_min ? speeds[i] : speed_min;
		speed_max = speeds[i]>=speed_max ? speeds[i] : speed_max;
	}

	//get image size
	int x_size = floor((x_max-x_min)+100);
	int y_size = floor((y_max-y_min)+100);

	for(int i=0;i<points.size();i++)
	{
		points[i].x = points[i].x-x_min+50;
		points[i].y = points[i].y-y_min+50;

		std::cout<<points[i].x<<"\t"<<points[i].y<<"\t"<<speeds[i]<<std::endl;
	}


	Mat image = Mat(y_size,x_size,CV_8UC3,cv::Scalar(255,255,255));

	//draw spot
	for(int i=0;i<points.size();i++)
	{
		cv::circle(image,points[i],3,get_Color_By_Speed(speeds[i],speed_min,speed_max),5);  
	}

	//draw trace
	for(int i=0;i<points.size()-1;i++)
	{
		cv::line(image,points[i],points[i+1], cv::Scalar(0, 0, 0),1);  
	}

	cv::imwrite(car_data[car_id].car_id_+".jpg",image);

	return true;

}

//draw lots of car together
bool draw_Cars_Trace(const std::vector<Trace_Data> &trace_data,const std::vector<Car_Data> &car_data,const std::vector<int> &car_ids)
{
	double x_min,x_max,y_min,y_max;
	double speed_min,speed_max;

	for(int j=0;j<car_ids.size();j++)
	{
		int car_id = car_ids[j];

		for(int i=0;i<car_data[car_id].car_statuses_id_.size();i++)
		{

			cv::Point2d point_temp;
			point_temp.x = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.x_;
			point_temp.y = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.y_;
			double speed = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.speed_;

			//std::cout<<point_temp.x<<"\t"<<point_temp.y<<std::endl;

			if(i==0 && j==0)
			{
				x_min = point_temp.x;
				x_max = point_temp.x;
				y_min = point_temp.y;
				y_max = point_temp.y;
				speed_min = speed;
				speed_max = speed;

				continue;
			}
			x_min = point_temp.x< x_min ? point_temp.x : x_min;
			x_max = point_temp.x>=x_max ? point_temp.x : x_max;
			y_min = point_temp.y< y_min ? point_temp.y : y_min;
			y_max = point_temp.y>=y_max ? point_temp.y : y_max;
			speed_min = speed< speed_min ? speed : speed_min;
			speed_max = speed>=speed_max ? speed : speed_max;
		}


	}


	//get image size
	int x_size = floor((x_max-x_min)+100);
	int y_size = floor((y_max-y_min)+100);

	Mat image = Mat(y_size,x_size,CV_8UC3,cv::Scalar(255,255,255));

	for(int j=0;j<car_ids.size();j++)
	{
		int car_id = car_ids[j];
		std::vector<cv::Point2d> points;
		std::vector<double> speeds;

		for(int i=0;i<car_data[car_id].car_statuses_id_.size();i++)
		{

			cv::Point2d point_temp;
			point_temp.x = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.x_- x_min + 50;
			point_temp.y = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.y_ - y_min + 50;
			double speed = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.speed_;

			points.push_back(point_temp);
			speeds.push_back(speed);
			//std::cout<<point_temp.x<<"\t"<<point_temp.y<<std::endl;
		}

		//draw spot
		for(int i=0;i<points.size();i++)
		{
			cv::circle(image,points[i],2,get_Color_By_Speed(speeds[i],speed_min,speed_max),2);  
		}

		//draw trace
		for(int i=0;i<points.size()-1;i++)
		{
			cv::line(image,points[i],points[i+1], cv::Scalar(0, 0, 0),1);  
		}

	}

	std::string image_name="";
	for(int j=0;j<car_ids.size();j++)
	{
		image_name = image_name + "_" + car_data[j].car_id_;
	}

	cv::imwrite(image_name+".jpg",image);

	return true;

}

bool draw_Anim_Trace(const std::vector<Trace_Data> &trace_data,std::vector<Car_Data> &car_data)
{
	double min_time_global,max_time_global;

	//get the max time and min time
	//get each car's life time
	for(int j=0;j<car_data.size();j++)
	{	
		double min_time_local,max_time_local;
		for(int i=0;i<car_data[j].car_statuses_id_.size();i++)
		{	
			double current_time = trace_data[car_data[j].car_statuses_id_[i]].car_status_.time_;
			if(i==0)
			{
				min_time_local = current_time;
				max_time_local = current_time;
				continue;
			}
			min_time_local = current_time< min_time_local ? current_time : min_time_local;
			max_time_local = current_time>=max_time_local ? current_time : max_time_local;
		}
		car_data[j].active_time_start_ = min_time_local;
		car_data[j].active_time_end_   = max_time_local;
		if(j==0)
		{
			min_time_global = min_time_local;
			max_time_global = max_time_global;
			continue;
		}
		min_time_global = min_time_local < min_time_global ?  min_time_local : min_time_global;
		max_time_global = max_time_local>=max_time_global ? max_time_local : max_time_global;
	}


	std::cout<<min_time_global<<std::endl;
	std::cout<<max_time_global<<std::endl;

	//get the video frame's size
	double x_min,x_max,y_min,y_max;
	double speed_min,speed_max;
	for(int j=0;j<car_data.size();j++)
	{
		int car_id = j;
		for(int i=0;i<car_data[car_id].car_statuses_id_.size();i++)
		{
			cv::Point2d point_temp;
			point_temp.x = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.x_;
			point_temp.y = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.y_;
			double speed = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.speed_;

			//std::cout<<point_temp.x<<"\t"<<point_temp.y<<std::endl;

			if(i==0 && j==0)
			{
				x_min = point_temp.x;
				x_max = point_temp.x;
				y_min = point_temp.y;
				y_max = point_temp.y;
				speed_min = speed;
				speed_max = speed;

				continue;
			}
			x_min = point_temp.x< x_min ? point_temp.x : x_min;
			x_max = point_temp.x>=x_max ? point_temp.x : x_max;
			y_min = point_temp.y< y_min ? point_temp.y : y_min;
			y_max = point_temp.y>=y_max ? point_temp.y : y_max;
			speed_min = speed< speed_min ? speed : speed_min;
			speed_max = speed>=speed_max ? speed : speed_max;
		}
	}

	//get frame size
	int x_size = floor((x_max-x_min)+100);
	int y_size = floor((y_max-y_min)+100);

	//get frame number
	int frame_count = floor((max_time_global-min_time_global)/1);

	double rate = 25;//视频的帧率  
    Size videoSize(x_size,y_size);  
    VideoWriter writer("VideoTest.avi", CV_FOURCC('X', 'V', 'I', 'D'), rate, videoSize);  

	for(int time=0;time<frame_count;time++)
	{
		double current_time = (time*1+min_time_global);

		//for each car first check it's active time
		//then check each point
		Mat image = Mat(y_size,x_size,CV_8UC3,cv::Scalar(255,255,255));

		for(int j=0;j<car_data.size();j++)
		{
			//check time
			if(car_data[j].active_time_end_<current_time || car_data[j].active_time_start_>current_time)
			{
				continue;
			}

			int car_id = j;

			std::vector<cv::Point2d> points;
			std::vector<double> speeds;

			for(int i=0;i<car_data[car_id].car_statuses_id_.size();i++)
			{
				double point_time = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.time_;
				if(point_time>current_time)
				{
					continue;
				}

				cv::Point2d point_temp;
				point_temp.x = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.x_- x_min + 50;
				point_temp.y = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.y_ - y_min + 50;
				double speed = trace_data[car_data[car_id].car_statuses_id_[i]].car_status_.speed_;
				
				//add point by time
				points.push_back(point_temp);
				speeds.push_back(speed);

				//std::cout<<point_temp.x<<"\t"<<point_temp.y<<std::endl;
			}

			//draw spot
			for(int i=0;i<points.size();i++)
			{
				cv::circle(image,points[i],2,get_Color_By_Speed(speeds[i],speed_min,speed_max),2);  
			}

			//draw trace
			for(int i=0;i<points.size()-1;i++)
			{
				cv::line(image,points[i],points[i+1], cv::Scalar(0, 0, 0),1);  
			}

		}

		//imwrite(s+".jpg",image);
		//add frame
		writer<<image;

	}

	//save image
	writer.release();

	return true;
}

int main(int argc,char *argv[])
{	
	if(argc<2)
	{
		std::cerr<<"Not enough parameter"<<std::endl;
		return -1;
	}

	//trace file path
	std::string trace_file_path = argv[1];

	//read data from file
	std::vector<Trace_Data> trace_data;
	read_Data_From_File(trace_file_path,trace_data);

	//divide into car
	std::vector<Car_Data> car_data;
	get_Each_Car_Data(trace_data,car_data);

	//draw one car
	//draw_Car_Trace(trace_data,car_data,0);
	//draw_Car_Trace(trace_data,car_data,1);

	//draw many cars
	std::vector<int> cars;
	cars.push_back(0);
	cars.push_back(1);
	cars.push_back(2);
	cars.push_back(3);
	cars.push_back(4);
	cars.push_back(5);
	cars.push_back(6);
	cars.push_back(7);
	//draw_Cars_Trace(trace_data,car_data,cars);

	//show video, means draw by time
	draw_Anim_Trace(trace_data,car_data);

	return 0;
}
