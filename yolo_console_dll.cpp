#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/types.h>

bool 	use_GUI = true,	//false
	use_fps = true;	//false

// Tambahan
int	A, B, C, D,
	ball_x, ball_y, ball_w, ball_h,

	gX_under[2], gY_under[2], gW_under[2], gH_under[2],//aslinya
	//gX_under, gY_under, gW_under, gH_under,
	goal_x[2][2], goal_y[2][2], goal_w[2][2], goal_h[2][2], goal_cx, goal_cy,//aslinya
	//goal_x, goal_y, goal_w, goal_h, goal_cx, goal_cy,

	pinalty_x, pinalty_y, pinalty_w, pinalty_h,
	L_x[2], L_y[2], L_w[2], L_h[2],
	Lcross_x[2][2], Lcross_y[2][2], Lcross_w[2][2], Lcross_h[2][2],
	X_x[2], X_y[2], X_w[2], X_h[2],
	Xcross_x[2][2], Xcross_y[2][2], Xcross_w[2][2], Xcross_h[2][2],
	T_x[2], T_y[2], T_w[2], T_h[2],
	Tcross_x[2][2], Tcross_y[2][2], Tcross_w[2][2], Tcross_h[2][2];

float	ball_d, gD_under[2], goal_d[2][2], pinalty_d, L_d[2], Lcross_d[2][2], X_d[2], Xcross_d[2][2], T_d[2], Tcross_d[2][2];
//float	ball_d, gD_under, goal_d, pinalty_d, L_d[2], Lcross_d[2][2], X_d[2], Xcross_d[2][2], T_d[2], Tcross_d[2][2];

//UDP LAN
/*#define BUFLEN 4096
#define PORT 2000
#define SERVER_IP "192.168.123.11"

int viSoc;
struct sockaddr_in addrVision;
char str [BUFLEN];

void initSoc(){
    viSoc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    memset((char *) &addrVision, 0, sizeof(addrVision));
    addrVision.sin_family = AF_INET;
    addrVision.sin_port = htons(PORT);
    inet_aton(SERVER_IP, &addrVision.sin_addr);
}*/

//Localhost
#define	BUFLEN	      	 4096
#define OUT_PORT_DARKNET 2000
struct sockaddr_in si_me, si_other, Remote;
int s, slen = sizeof(si_other), recv_len;
int sm, lm , im, c;
char str [BUFLEN];

void die(char* s){
	perror(s);
    	exit(1);
}

// It makes sense only for video-Camera (not for video-File)
// To use - uncomment the following line. Optical-flow is supported only by OpenCV 3.x - 4.x
//#define TRACK_OPTFLOW
//#define GPU

// To use 3D-stereo camera ZED - uncomment the following line. ZED_SDK should be installed.
//#define ZED_STEREO


#include "yolo_v2_class.hpp"    // imported functions from DLL

#ifdef OPENCV
#ifdef ZED_STEREO
#include <sl/Camera.hpp>
#if ZED_SDK_MAJOR_VERSION == 2
#define ZED_STEREO_2_COMPAT_MODE
#endif

#undef GPU // avoid conflict with sl::MEM::GPU

#ifdef ZED_STEREO_2_COMPAT_MODE
#pragma comment(lib, "sl_core64.lib")
#pragma comment(lib, "sl_input64.lib")
#endif
#pragma comment(lib, "sl_zed64.lib")

float getMedian(std::vector<float> &v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
    bool valid_measure;
    int i, j;
    const unsigned int R_max_global = 10;

    std::vector<bbox_t> bbox3d_vect;

    for (auto &cur_box : bbox_vect) {

        const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
        const unsigned int R_max = std::min(R_max_global, obj_size / 2);
        int center_i = cur_box.x + cur_box.w * 0.5f, center_j = cur_box.y + cur_box.h * 0.5f;

        std::vector<float> x_vect, y_vect, z_vect;
        for (int R = 0; R < R_max; R++) {
            for (int y = -R; y <= R; y++) {
                for (int x = -R; x <= R; x++) {
                    i = center_i + x;
                    j = center_j + y;
                    sl::float4 out(NAN, NAN, NAN, NAN);
                    if (i >= 0 && i < xyzrgba.cols && j >= 0 && j < xyzrgba.rows) {
                        cv::Vec4f &elem = xyzrgba.at<cv::Vec4f>(j, i);  // x,y,z,w
                        out.x = elem[0];
                        out.y = elem[1];
                        out.z = elem[2];
                        out.w = elem[3];
                    }
                    valid_measure = std::isfinite(out.z);
                    if (valid_measure)
                    {
                        x_vect.push_back(out.x);
                        y_vect.push_back(out.y);
                        z_vect.push_back(out.z);
                    }
                }
            }
        }

        if (x_vect.size() * y_vect.size() * z_vect.size() > 0)
        {
            cur_box.x_3d = getMedian(x_vect);
            cur_box.y_3d = getMedian(y_vect);
            cur_box.z_3d = getMedian(z_vect);
        }
        else {
            cur_box.x_3d = NAN;
            cur_box.y_3d = NAN;
            cur_box.z_3d = NAN;
        }

        bbox3d_vect.emplace_back(cur_box);
    }

    return bbox3d_vect;
}

cv::Mat slMat2cvMat(sl::Mat &input) {
    int cv_type = -1; // Mapping between MAT_TYPE and CV_TYPE
    if(input.getDataType() ==
#ifdef ZED_STEREO_2_COMPAT_MODE
        sl::MAT_TYPE_32F_C4
#else
        sl::MAT_TYPE::F32_C4
#endif
        ) {
        cv_type = CV_32FC4;
    } else cv_type = CV_8UC4; // sl::Mat used are either RGBA images or XYZ (4C) point clouds
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(
#ifdef ZED_STEREO_2_COMPAT_MODE
        sl::MEM::MEM_CPU
#else
        sl::MEM::CPU
#endif
        ));
}

cv::Mat zed_capture_rgb(sl::Camera &zed) {
    sl::Mat left;
    zed.retrieveImage(left);
    cv::Mat left_rgb;
    cv::cvtColor(slMat2cvMat(left), left_rgb, CV_RGBA2RGB);
    return left_rgb;
}

cv::Mat zed_capture_3d(sl::Camera &zed) {
    sl::Mat cur_cloud;
    zed.retrieveMeasure(cur_cloud,
#ifdef ZED_STEREO_2_COMPAT_MODE
        sl::MEASURE_XYZ
#else
        sl::MEASURE::XYZ
#endif
        );
    return slMat2cvMat(cur_cloud).clone();
}

static sl::Camera zed; // ZED-camera

#else   // ZED_STEREO
std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
    return bbox_vect;
}
#endif  // ZED_STEREO


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
/*
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
*/
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
	    obj_name += " - " + std::to_string((int)(i.prob * 100)) + "%";
            //if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}
#endif    // OPENCV

// Tambahan
void get_object_coordinate(std::vector<bbox_t> const result_vec, int current_det_fps = -1, int current_cap_fps = -1) {
    for (auto &i : result_vec) {
	if(i.obj_id == 0) { //bola
		ball_x = (i.w / 2) + i.x;
		ball_y = (i.h / 2) + i.y;
		ball_w = i.w;
		ball_h = i.h;
		
		ball_d = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
		}

	if(i.obj_id == 1) { //goal
		if (A > 0) {//aslinya
		//	gX_under = (i.w / 2) + i.x;
		//	gY_under = (i.h / 2) + i.y;
		//	gW_under = i.w;
		//	gH_under = i.h;
		//	gD_under = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;


			gX_under[1] = (i.w / 2) + i.x;							//aslinya
			gY_under[1] = (i.h / 2) + i.y;							//aslinya
			gW_under[1] = i.w;								//aslinya
			gH_under[1] = i.h;								//aslinya
			gD_under[1] = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;	//aslinya
			//printf("\nkanan");
		} else {										//aslinya
			gX_under[0] = (i.w / 2) + i.x;							//aslinya
			gY_under[0] = (i.h / 2) + i.y;							//aslinya
			gW_under[0] = i.w;								//aslinya
			gH_under[0] = i.h;								//aslinya
			gD_under[0] = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;	//aslinya
			printf("\nkiri");
			A = A + 1;									//aslinya
		}

		if (gX_under[0] != -1 && gX_under[1] != -1) {						//aslinya
			goal_cx = (gX_under[0] + gX_under[1]) / 2;					//aslinya
			goal_cy = (gY_under[0] + gY_under[1]) / 2;					//aslinya
		}											//aslinya

		//goal_cx = (gX_under + gX_under)/2;
		//goal_cy = (gY_under + gY_under)/2;
	}
/*
	if(i.obj_id == 3) { //pinalty_point
		pinalty_x = (i.w / 2) + i.x;
		pinalty_y = (i.h / 2) + i.y;
		pinalty_w = i.w;
		pinalty_h = i.h;
		pinalty_d = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
	}
*/
	if(i.obj_id == 2) { //L Cross
		if (B > 0) {
			L_x[1] = (i.w / 2) + i.x;
			L_y[1]  = (i.h / 2) + i.y;
			L_w[1]  = i.w;
			L_h[1]  = i.h;
			L_d[1]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			//printf("\nkanan");
		} else {
			L_x[0] = (i.w / 2) + i.x;
			L_y[0]  = (i.h / 2) + i.y;
			L_w[0]  = i.w;
			L_h[0]  = i.h;
			L_d[0]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			printf("\nkiri");
			B = B + 1;
		}
	}
	if(i.obj_id == 3) { //X Cross
		if (C > 0) {
			X_x[1] = (i.w / 2) + i.x;
			X_y[1]  = (i.h / 2) + i.y;
			X_w[1]  = i.w;
			X_h[1]  = i.h;
			X_d[1]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			//printf("\nkanan");
		} else {
			X_x[0] = (i.w / 2) + i.x;
			X_y[0]  = (i.h / 2) + i.y;
			X_w[0]  = i.w;
			X_h[0]  = i.h;
			X_d[0]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			//printf("\nkiri");
			C = C + 1;
		}
	}
/*	if(i.obj_id == 6) { //T Cross
		if (D > 0) {
			T_x[1] = (i.w / 2) + i.x;
			T_y[1]  = (i.h / 2) + i.y;
			T_w[1]  = i.w;
			T_h[1]  = i.h;
			T_d[1]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			//printf("\nkanan");
		} else {
			T_x[0] = (i.w / 2) + i.x;
			T_y[0]  = (i.h / 2) + i.y;
			T_w[0]  = i.w;
			T_h[0]  = i.h;
			T_d[0]  = sqrt(pow(i.x_3d,2) + pow(i.y_3d,2) + pow(i.z_3d,2)) * 100;
			//printf("\nkiri");
			D = D + 1;
		}
	}
*/    }
    //goal
    if (gX_under[0] < gX_under[1]) {	//aslinya
	goal_x[0][0] = gX_under[0];	//aslinya
	goal_y[0][0] = gY_under[0];	//aslinya
	goal_w[0][0] = gW_under[0];	//aslinya
	goal_h[0][0] = gH_under[0];	//aslinya
	goal_d[0][0] = gD_under[0];	//aslinya

	goal_x[0][1] = gX_under[1];	//aslinya
	goal_y[0][1] = gY_under[1];	//aslinya
	goal_w[0][1] = gW_under[1];	//aslinya
	goal_h[0][1] = gH_under[1];	//aslinya
	goal_d[0][1] = gD_under[1];	//aslinya
    } else {
	goal_x[0][0] = gX_under[1];	//aslinya
	goal_y[0][0] = gY_under[1];	//aslinya
	goal_w[0][0] = gW_under[1];	//aslinya
	goal_h[0][0] = gH_under[1];	//aslinya
	goal_d[0][0] = gD_under[1];	//aslinya

	goal_x[0][1] = gX_under[0];	//aslinya
	goal_y[0][1] = gY_under[0];	//aslinya
	goal_w[0][1] = gW_under[0];	//aslinya
	goal_h[0][1] = gH_under[0];	//aslinya
	goal_d[0][1] = gD_under[0];	//aslinya
    }


    //L Cross
    if (L_x[0] < L_x[1]) {
	Lcross_x[0][0] = L_x[0];
	Lcross_y[0][0] = L_y[0];
	Lcross_w[0][0] = L_w[0];
	Lcross_h[0][0] = L_h[0];
	Lcross_d[0][0] = L_d[0];
	Lcross_x[0][1] = L_x[1];
	Lcross_y[0][1] = L_y[1];
	Lcross_w[0][1] = L_w[1];
	Lcross_h[0][1] = L_h[1];
	Lcross_d[0][1] = L_d[1];
    } else {
	Lcross_x[0][0] = L_x[1];
	Lcross_y[0][0] = L_y[1];
	Lcross_w[0][0] = L_w[1];
	Lcross_h[0][0] = L_h[1];
	Lcross_d[0][0] = L_d[1];
	Lcross_x[0][1] = L_x[0];
	Lcross_y[0][1] = L_y[0];
	Lcross_w[0][1] = L_w[0];
	Lcross_h[0][1] = L_h[0];
	Lcross_d[0][1] = L_d[0];
    }
    //X Cross
    if (X_x[0] < X_x[1]) {
	Xcross_x[0][0] = X_x[0];
	Xcross_y[0][0] = X_y[0];
	Xcross_w[0][0] = X_w[0];
	Xcross_h[0][0] = X_h[0];
	Xcross_d[0][0] = X_d[0];
	Xcross_x[0][1] = X_x[1];
	Xcross_y[0][1] = X_y[1];
	Xcross_w[0][1] = X_w[1];
	Xcross_h[0][1] = X_h[1];
	Xcross_d[0][1] = X_d[1];
    } else {
	Xcross_x[0][0] = X_x[1];
	Xcross_y[0][0] = X_y[1];
	Xcross_w[0][0] = X_w[1];
	Xcross_h[0][0] = X_h[1];
	Xcross_d[0][0] = X_d[1];
	Xcross_x[0][1] = X_x[0];
	Xcross_y[0][1] = X_y[0];
	Xcross_w[0][1] = X_w[0];
	Xcross_h[0][1] = X_h[0];
	Xcross_d[0][1] = X_d[0];
    }
/*
    //T Cross
    if (T_x[0] < T_x[1]) {
	Tcross_x[0][0] = T_x[0];
	Tcross_y[0][0] = T_y[0];
	Tcross_w[0][0] = T_w[0];
	Tcross_h[0][0] = T_h[0];
	Tcross_d[0][0] = T_d[0];
	Tcross_x[0][1] = T_x[1];
	Tcross_y[0][1] = T_y[1];
	Tcross_w[0][1] = T_w[1];
	Tcross_h[0][1] = T_h[1];
	Tcross_d[0][1] = T_d[1];
    } else {
	Tcross_x[0][0] = T_x[1];
	Tcross_y[0][0] = T_y[1];
	Tcross_w[0][0] = T_w[1];
	Tcross_h[0][0] = T_h[1];
	Tcross_d[0][0] = T_d[1];
	Tcross_x[0][1] = T_x[0];
	Tcross_y[0][1] = T_y[0];
	Tcross_w[0][1] = T_w[0];
	Tcross_h[0][1] = T_h[0];
	Tcross_d[0][1] = T_d[0];
    }
*/
    //goal
    if (goal_x[0][0] != -1 && goal_x[0][1] != -1) {				//aslinya
	goal_x[1][0] = goal_x[0][0];						//aslinya
	goal_y[1][0] = goal_y[0][0];						//aslinya
	goal_w[1][0] = goal_w[0][0];						//aslinya
	goal_h[1][0] = goal_h[0][0];						//aslinya
	goal_d[1][0] = goal_d[0][0];						//aslinya

	goal_x[1][1] = goal_x[0][1];						//aslinya
	goal_y[1][1] = goal_y[0][1];						//aslinya
	goal_w[1][1] = goal_w[0][1];						//aslinya
	goal_h[1][1] = goal_h[0][1];						//aslinya
	goal_d[1][1] = goal_d[0][1];						//aslinya
    } else if (goal_x[0][0] == -1 || goal_x[0][1] == -1){			//aslinya
	if ((goal_x[0][1] > Lcross_x[0][1]) && Lcross_x[0][1] != -1) {		//aslinya
		goal_x[1][0] = goal_x[0][1];					//aslinya
		goal_y[1][0] = goal_y[0][1];					//aslinya
		goal_w[1][0] = goal_w[0][1];					//aslinya
		goal_h[1][0] = goal_h[0][1];					//aslinya
		goal_d[1][0] = goal_d[0][1];					//aslinya
	} else if ((goal_x[0][1] < Lcross_x[0][1]) && Lcross_x[0][1] != -1) {	//aslinya
		goal_x[1][1] = goal_x[0][1];					//aslinya
		goal_y[1][1] = goal_y[0][1];					//aslinya
		goal_w[1][1] = goal_w[0][1];					//aslinya
		goal_h[1][1] = goal_h[0][1];					//aslinya
		goal_d[1][1] = goal_d[0][1];					//aslinya
	}
    }
    //L Cross
    if (Lcross_x[0][0] != -1 && Lcross_x[0][0] != -1) {
	Lcross_x[1][0] = Lcross_x[0][0];
	Lcross_y[1][0] = Lcross_y[0][0];
	Lcross_w[1][0] = Lcross_w[0][0];
	Lcross_h[1][0] = Lcross_h[0][0];
	Lcross_d[1][0] = Lcross_d[0][0];
	Lcross_x[1][1] = Lcross_x[0][1];
	Lcross_y[1][1] = Lcross_y[0][1];
	Lcross_w[1][1] = Lcross_w[0][1];
	Lcross_h[1][1] = Lcross_h[0][1];
	Lcross_d[1][1] = Lcross_d[0][1];
    } else if (Lcross_x[0][0] == -1 || Lcross_x[0][0] == -1) {
	if (Lcross_x[0][1] < goal_x[0][1]) {
//	if (Lcross_x[0][1] < goal_x) {
		Lcross_x[1][0] = Lcross_x[0][1];
		Lcross_y[1][0] = Lcross_y[0][1];
		Lcross_w[1][0] = Lcross_w[0][1];
		Lcross_h[1][0] = Lcross_h[0][1];
		Lcross_d[1][0] = Lcross_d[0][1];
	} else {
		Lcross_x[1][1] = Lcross_x[0][1];
		Lcross_y[1][1] = Lcross_y[0][1];
		Lcross_w[1][1] = Lcross_w[0][1];
		Lcross_h[1][1] = Lcross_h[0][1];
		Lcross_d[1][1] = Lcross_d[0][1];
	}
    }
    //X Cross
    if (Xcross_x[0][0] != -1 && Xcross_x[0][0] != -1) {
	Xcross_x[1][0] = Xcross_x[0][0];
	Xcross_y[1][0] = Xcross_y[0][0];
	Xcross_w[1][0] = Xcross_w[0][0];
	Xcross_h[1][0] = Xcross_h[0][0];
	Xcross_d[1][0] = Xcross_d[0][0];
	Xcross_x[1][1] = Xcross_x[0][1];
	Xcross_y[1][1] = Xcross_y[0][1];
	Xcross_w[1][1] = Xcross_w[0][1];
	Xcross_h[1][1] = Xcross_h[0][1];
	Xcross_d[1][1] = Xcross_d[0][1];
    } else if (Xcross_x[0][0] == -1 || Xcross_x[0][0] == -1) {
	if ((Xcross_x[0][1] > Lcross_x[0][1]) && Lcross_x[0][1] != -1) {
		Xcross_x[1][0] = Xcross_x[0][1];
		Xcross_y[1][0] = Xcross_y[0][1];
		Xcross_w[1][0] = Xcross_w[0][1];
		Xcross_h[1][0] = Xcross_h[0][1];
		Xcross_d[1][0] = Xcross_d[0][1];
	} else if ((Xcross_x[0][1] < Lcross_x[0][1]) && Lcross_x[0][1] != -1) {
		Xcross_x[1][1] = Xcross_x[0][1];
		Xcross_y[1][1] = Xcross_y[0][1];
		Xcross_w[1][1] = Xcross_w[0][1];
		Xcross_h[1][1] = Xcross_h[0][1];
		Xcross_d[1][1] = Xcross_d[0][1];
	}
    }
	/*
    //T Cross
    if (Tcross_x[0][0] != -1 && Tcross_x[0][0] != -1) {
	Tcross_x[1][0] = Tcross_x[0][0];
	Tcross_y[1][0] = Tcross_y[0][0];
	Tcross_w[1][0] = Tcross_w[0][0];
	Tcross_h[1][0] = Tcross_h[0][0];
	Tcross_d[1][0] = Tcross_d[0][0];
	Tcross_x[1][1] = Tcross_x[0][1];
	Tcross_y[1][1] = Tcross_y[0][1];
	Tcross_w[1][1] = Tcross_w[0][1];
	Tcross_h[1][1] = Tcross_h[0][1];
	Tcross_d[1][1] = Tcross_d[0][1];
    } else if (Tcross_x[0][0] == -1 || Tcross_x[0][0] == -1) {
	if (Tcross_x[0][1] < Xcross_x[0][1]) {
		Tcross_x[1][0] = Tcross_x[0][1];
		Tcross_y[1][0] = Tcross_y[0][1];
		Tcross_w[1][0] = Tcross_w[0][1];
		Tcross_h[1][0] = Tcross_h[0][1];
		Tcross_d[1][0] = Tcross_d[0][1];
	} else {
		Tcross_x[1][1] = Tcross_x[0][1];
		Tcross_y[1][1] = Tcross_y[0][1];
		Tcross_w[1][1] = Tcross_w[0][1];
		Tcross_h[1][1] = Tcross_h[0][1];
		Tcross_d[1][1] = Tcross_d[0][1];
	}
    }
	*/

    printf("\033[2J");
    printf("\033[1;1H");
    if (use_fps) { printf("\nFPS detection: %d, FPS capture: %d", current_det_fps, current_cap_fps); }

    //printf("\nFPS detection: %d, FPS Capture: %d", current_det_fps, current_cap_fps);


    printf("\nBall Coordinat : %d , %d\n", ball_x, ball_y);
    printf("Ball Distance : %.2f \n", ball_d);
//    printf("\niwball = %d", ball_w);
//    printf("\nihball = %d", ball_h);
//    printf("\nixball = %d", ball_ix);
//    printf("\niyball = %d", ball_iy);	
//    printf("\nix3dball = %d", ball_ix3);
//    printf("\niy3dball = %d", ball_iy3);
//    printf("\niz3dball = %d", ball_iz3);
//    printf("\nA = %d", A);


    printf("\nCenter Goal Coordinat : %d , %d\n", goal_cx, goal_cy);		//aslinya
    printf("Left Goal Coordinat : %d , %d\n", goal_x[1][0], goal_y[1][0]);	//aslinya
    printf("Left Goal Distance : %.2f \n", goal_d[1][0]);			//aslinya
    printf("Right Goal Coordinat : %d , %d\n", goal_x[1][1], goal_y[1][1]);	//aslinya
    printf("Right Goal Distance : %.2f \n", goal_d[1][1]);			//aslinya
    printf("\nCenter Goal Coordinat : %d , %d\n", goal_cx, goal_cy);
    printf("\nGoal Coordinat : %d , %d\n",gX_under , gY_under);
    printf("Goal Distance : %.2f \n", goal_d);




//    printf("\nPinalty Coordinat : %d , %d\n", pinalty_x, pinalty_y);
//    printf("Pinalty Distance : %.2f \n", pinalty_d);

    printf("\nLeft L_Cross Coordinat : %d , %d\n", Lcross_x[1][0], Lcross_y[1][0]);
    printf("Left L_Cross Distance : %.2f \n", Lcross_d[1][0]);
    printf("Right L_Cross Coordinat : %d , %d\n", Lcross_x[1][1], Lcross_y[1][1]);
    printf("Right L_Cross Distance : %.2f \n", Lcross_d[1][1]);

    printf("\nLeft X_Cross Coordinat : %d , %d\n", Xcross_x[1][0], Xcross_y[1][0]);
    printf("Left X_Cross Distance : %.2f \n", Xcross_d[1][0]);
    printf("Right X_Cross Coordinat : %d , %d\n", Xcross_x[1][1], Xcross_y[1][1]);
    printf("Right X_Cross Distance : %.2f \n", Xcross_d[1][1]);

    //printf("\nLeft T_Cross Coordinat : %d , %d\n", Tcross_x[1][0], Tcross_y[1][0]);
    //printf("Left T_Cross Distance : %.2f \n", Tcross_d[1][0]);
    //printf("Right T_Cross Coordinat : %d , %d\n", Tcross_x[1][1], Tcross_y[1][1]);
    //printf("Right T_Cross Distance : %.2f \n", Tcross_d[1][1]);
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

int main(int argc, char *argv[])
{
    std::string  names_file = "data/coco.names";
    std::string  cfg_file = "cfg/yolov3.cfg";
    std::string  weights_file = "yolov3.weights";
    std::string filename;

    if (argc > 4) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
        filename = argv[4];
    }
    else if (argc > 1) filename = argv[1];

    float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.2;

    Detector detector(cfg_file, weights_file);

    auto obj_names = objects_names_from_file(names_file);
    std::string out_videofile = "result.avi";
    bool const save_output_videofile = false;   // true - for history
    bool const send_network = false;        // true - for remote detection
    bool const use_kalman_filter = false;   // true - for stationary camera

    bool detection_sync = true;             // true - for video-file
#ifdef TRACK_OPTFLOW    // for slow GPU
    detection_sync = false;
    Tracker_optflow tracker_flow;
    //detector.wait_stream = true;
#endif  // TRACK_OPTFLOW

// Tambahan
    if(use_GUI) {
	cv::namedWindow("BarelangFC - Vision", CV_WINDOW_NORMAL);
	cv::moveWindow("BarelangFC - Vision", 0, 0);
	cv::resizeWindow("BarelangFC - Vision", 640, 360); //Webcam
	//cv::resizeWindow("BarelangFC - Vision", 672, 376);
	//cv::resizeWindow("BarelangFC - Vision", 1280, 720);
    }


    //connect socket udp LAN
    //initSoc();

    //connect socket localhost
    if ((sm=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)die("socket");
    if ((lm=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)die("socket");
    memset((char *) &si_me, 0, sizeof(si_me));
    memset((char *) &Remote, 0, sizeof(Remote));
    Remote.sin_family = AF_INET;
    Remote.sin_port = htons(OUT_PORT_DARKNET);
    if (inet_aton("127.0.0.1" , &Remote.sin_addr) == 0){
	fprintf(stderr, "inet_aton() failed\n");
	exit(1);
    }

    while (true)
    {
        std::cout << "input image or video filename: ";
        if(filename.size() == 0) std::cin >> filename;
        if (filename.size() == 0) break;

        try {
#ifdef OPENCV
            preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
            bool show_small_boxes = false;

            std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
            std::string const protocol = filename.substr(0, 7);
            if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" ||     // video file
                protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/" ||    // video network stream
                filename == "zed_camera" || file_ext == "svo" || filename == "web_camera")   // ZED stereo camera

            {
                if (protocol == "rtsp://" || protocol == "http://" || protocol == "https:/" || filename == "zed_camera" || filename == "web_camera")
                    detection_sync = false;

                cv::Mat cur_frame;
                std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
                std::atomic<int> current_fps_cap(0), current_fps_det(0);
                std::atomic<bool> exit_flag(false);
                std::chrono::steady_clock::time_point steady_start, steady_end;
                int video_fps = 25;
                bool use_zed_camera = false;

                track_kalman_t track_kalman;

#ifdef ZED_STEREO
                sl::InitParameters init_params;
                init_params.depth_minimum_distance = 0.5;
    #ifdef ZED_STEREO_2_COMPAT_MODE
                init_params.depth_mode = sl::DEPTH_MODE_ULTRA;
                init_params.camera_resolution = sl::RESOLUTION_HD720;// sl::RESOLUTION_HD1080, sl::RESOLUTION_HD720
                init_params.coordinate_units = sl::UNIT_METER;
                init_params.camera_buffer_count_linux = 2;
                if (file_ext == "svo") init_params.svo_input_filename.set(filename.c_str());
    #else
                init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
                init_params.camera_resolution = sl::RESOLUTION::HD720;// sl::RESOLUTION::HD1080, sl::RESOLUTION::HD720
                init_params.coordinate_units = sl::UNIT::METER;
                if (file_ext == "svo") init_params.input.setFromSVOFile(filename.c_str());
    #endif
                //init_params.sdk_cuda_ctx = (CUcontext)detector.get_cuda_context();
                init_params.sdk_gpu_id = detector.cur_gpu_id;

                if (filename == "zed_camera" || file_ext == "svo") {
                    std::cout << "ZED 3D Camera " << zed.open(init_params) << std::endl;
                    if (!zed.isOpened()) {
                        std::cout << " Error: ZED Camera should be connected to USB 3.0. And ZED_SDK should be installed. \n";
                        getchar();
                        return 0;
                    }
                    cur_frame = zed_capture_rgb(zed);
                    use_zed_camera = true;
                }
#endif  // ZED_STEREO

                cv::VideoCapture cap;
                if (filename == "web_camera") {
                    cap.open(0);
					cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
					cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
                    cap >> cur_frame;
                } else if (!use_zed_camera) {
                    cap.open(filename);
                    cap >> cur_frame;
                }
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                video_fps = cap.get(CV_CAP_PROP_FPS);
#else
                video_fps = cap.get(cv::CAP_PROP_FPS);
#endif
                cv::Size const frame_size = cur_frame.size();
                //cv::Size const frame_size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
                std::cout << "\n Video size: " << frame_size << std::endl;

                cv::VideoWriter output_video;
                if (save_output_videofile)
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                    output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
#else
                    output_video.open(out_videofile, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
#endif

                struct detection_data_t {
                    cv::Mat cap_frame;
                    std::shared_ptr<image_t> det_image;
                    std::vector<bbox_t> result_vec;
                    cv::Mat draw_frame;
                    bool new_detection;
                    uint64_t frame_id;
                    bool exit_flag;
                    cv::Mat zed_cloud;
                    std::queue<cv::Mat> track_optflow_queue;
                    detection_data_t() : new_detection(false), exit_flag(false) {}
                };

                const bool sync = detection_sync; // sync data exchange
                send_one_replaceable_object_t<detection_data_t> cap2prepare(sync), cap2draw(sync),
                    prepare2detect(sync), detect2draw(sync), draw2show(sync), draw2write(sync), draw2net(sync);

                std::thread t_cap, t_prepare, t_detect, t_post, t_draw, t_write, t_network;

                // capture new video-frame
                if (t_cap.joinable()) t_cap.join();
                t_cap = std::thread([&]()
                {
                    uint64_t frame_id = 0;
                    detection_data_t detection_data;
                    do {
                        detection_data = detection_data_t();
#ifdef ZED_STEREO
                        if (use_zed_camera) {
                            while (zed.grab() !=
        #ifdef ZED_STEREO_2_COMPAT_MODE
                                sl::SUCCESS
        #else
                                sl::ERROR_CODE::SUCCESS
        #endif
                                ) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            detection_data.cap_frame = zed_capture_rgb(zed);
                            detection_data.zed_cloud = zed_capture_3d(zed);
                        }
                        else
#endif   // ZED_STEREO
                        {
                            cap >> detection_data.cap_frame;
                        }
                        fps_cap_counter++;
                        detection_data.frame_id = frame_id++;
                        if (detection_data.cap_frame.empty() || exit_flag) {
                            std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size() << std::endl;
                            detection_data.exit_flag = true;
                            detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
                        }

                        if (!detection_sync) {
                            cap2draw.send(detection_data);       // skip detection
                        }
                        cap2prepare.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_cap exit \n";
                });


                // pre-processing video frame (resize, convertion)
                t_prepare = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = cap2prepare.receive();

                        det_image = detector.mat_to_image_resize(detection_data.cap_frame);
                        detection_data.det_image = det_image;
                        prepare2detect.send(detection_data);    // detection

                    } while (!detection_data.exit_flag);
                    std::cout << " t_prepare exit \n";
                });


                // detection by Yolo
                if (t_detect.joinable()) t_detect.join();
                t_detect = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = prepare2detect.receive();
                        det_image = detection_data.det_image;
                        std::vector<bbox_t> result_vec;

                        if(det_image)
                            result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height, thresh, true);  // true
                        fps_det_counter++;
                        //std::this_thread::sleep_for(std::chrono::milliseconds(150));

                        detection_data.new_detection = true;
                        detection_data.result_vec = result_vec;
                        detect2draw.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_detect exit \n";
                });

                // draw rectangles (and track objects)
                t_draw = std::thread([&]()
                {
                    std::queue<cv::Mat> track_optflow_queue;
                    detection_data_t detection_data;
                    do {
			// Tambahan
			ball_x = ball_y = ball_w = ball_h = -1; //ball
			ball_d = 0; //ball

			goal_cx = goal_cy = -1; //goal center								//aslinya
			gX_under[0] = gY_under[0] = gW_under[0] = gH_under[0] = -1; //goal left				//aslinya
			gX_under[1] = gY_under[1] = gW_under[1] = gH_under[1] = -1; //goal right			//aslinya
			goal_x[0][0] = goal_y[0][0] = goal_w[0][0] = goal_h[0][0] = -1; //goal left			//aslinya
			goal_x[0][1] = goal_y[0][1] = goal_w[0][1] = goal_h[0][1] = -1; //goal right			//aslinya
			goal_x[1][0] = goal_y[1][0] = goal_w[1][0] = goal_h[1][0] = -1; //goal left			//aslinya
			goal_x[1][1] = goal_y[1][1] = goal_w[1][1] = goal_h[1][1] = -1; //goal right			//aslinya
			gD_under[0] = gD_under[1] = goal_d[0][0] = goal_d[0][1] = goal_d[1][0] = goal_d[1][1] = 0;	//aslinya


//			goal_cx = goal_cy = -1; //goal center
//			gX_under = gY_under = gW_under = gH_under = -1;
//			goal_x = goal_y = goal_w = goal_h = -1;
//			gD_under = 0;



			//pinalty_x = pinalty_y = pinalty_w = pinalty_h = -1; //pinalty
			//pinalty_d = 0; //pinalty
			L_x[0] = L_y[0] = L_w[0] = L_h[0] = -1; //L Cross
			L_x[1] = L_y[1] = L_w[1] = L_h[1] = -1; //L Cross
			Lcross_x[0][0] = Lcross_y[0][0] = Lcross_w[0][0] = Lcross_h[0][0] = -1; //L Cross
			Lcross_x[0][1] = Lcross_y[0][1] = Lcross_w[0][1] = Lcross_h[0][1] = -1; //L Cross
			Lcross_x[1][0] = Lcross_y[1][0] = Lcross_w[1][0] = Lcross_h[1][0] = -1; //L Cross
			Lcross_x[1][1] = Lcross_y[1][1] = Lcross_w[1][1] = Lcross_h[1][1] = -1; //L Cross
			L_d[0] = L_d[1] = Lcross_d[0][0] = Lcross_d[0][1] = Lcross_d[1][0] = Lcross_d[1][1] = 0;
			X_x[0] = X_y[0] = X_w[0] = X_h[0] = -1; //X Cross
			X_x[1] = X_y[1] = X_w[1] = X_h[1] = -1; //X Cross
			Xcross_x[0][0] = Xcross_y[0][0] = Xcross_w[0][0] = Xcross_h[0][0] = -1; //X Cross
			Xcross_x[0][1] = Xcross_y[0][1] = Xcross_w[0][1] = Xcross_h[0][1] = -1; //X Cross
			Xcross_x[1][0] = Xcross_y[1][0] = Xcross_w[1][0] = Xcross_h[1][0] = -1; //X Cross
			Xcross_x[1][1] = Xcross_y[1][1] = Xcross_w[1][1] = Xcross_h[1][1] = -1; //X Cross
			X_d[0] = X_d[1] = Xcross_d[0][0] = Xcross_d[0][1] = Xcross_d[1][0] = Xcross_d[1][1] = 0;
			//T_x[0] = T_y[0] = T_w[0] = T_h[0] = -1; //T Cross
			//T_x[1] = T_y[1] = T_w[1] = T_h[1] = -1; //T Cross
			//Tcross_x[0][0] = Tcross_y[0][0] = Tcross_w[0][0] = Tcross_h[0][0] = -1; //T Cross
			//Tcross_x[0][1] = Tcross_y[0][1] = Tcross_w[0][1] = Tcross_h[0][1] = -1; //T Cross
			//Tcross_x[1][0] = Tcross_y[1][0] = Tcross_w[1][0] = Tcross_h[1][0] = -1; //T Cross
			//Tcross_x[1][1] = Tcross_y[1][1] = Tcross_w[1][1] = Tcross_h[1][1] = -1; //T Cross
			//T_d[0] = T_d[1] = Tcross_d[0][0] = Tcross_d[0][1] = Tcross_d[1][0] = Tcross_d[1][1] = 0;
			A = B = C = D = 0;

                        // for Video-file
                        if (detection_sync) {
                            detection_data = detect2draw.receive();
                        }
                        // for Video-camera
                        else
                        {
                            // get new Detection result if present
                            if (detect2draw.is_object_present()) {
                                cv::Mat old_cap_frame = detection_data.cap_frame;   // use old captured frame
                                detection_data = detect2draw.receive();
                                if (!old_cap_frame.empty()) detection_data.cap_frame = old_cap_frame;
                            }
                            // get new Captured frame
                            else {
                                std::vector<bbox_t> old_result_vec = detection_data.result_vec; // use old detections
                                detection_data = cap2draw.receive();
                                detection_data.result_vec = old_result_vec;
                            }
                        }

                        cv::Mat cap_frame = detection_data.cap_frame;
                        cv::Mat draw_frame = detection_data.cap_frame.clone();
                        std::vector<bbox_t> result_vec = detection_data.result_vec;

#ifdef TRACK_OPTFLOW
                        if (detection_data.new_detection) {
                            tracker_flow.update_tracking_flow(detection_data.cap_frame, detection_data.result_vec);
                            while (track_optflow_queue.size() > 0) {
                                draw_frame = track_optflow_queue.back();
                                result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), false);
                                track_optflow_queue.pop();
                            }
                        }
                        else {
                            track_optflow_queue.push(cap_frame);
                            result_vec = tracker_flow.tracking_flow(cap_frame, false);
                        }
                        detection_data.new_detection = true;    // to correct kalman filter
#endif //TRACK_OPTFLOW

                        // track ID by using kalman filter
                        if (use_kalman_filter) {
                            if (detection_data.new_detection) {
                                result_vec = track_kalman.correct(result_vec);
                            }
                            else {
                                result_vec = track_kalman.predict();
                            }
                        }
                        // track ID by using custom function
                        else {
                            int frame_story = std::max(5, current_fps_cap.load());
                            result_vec = detector.tracking_id(result_vec, true, frame_story, 40);
                        }

                        if (use_zed_camera && !detection_data.zed_cloud.empty()) {
                            result_vec = get_3d_coordinates(result_vec, detection_data.zed_cloud);
                        }

                        //small_preview.set(draw_frame, result_vec);
                        //large_preview.set(draw_frame, result_vec);
                        draw_boxes(draw_frame, result_vec, obj_names, current_fps_det, current_fps_cap);
			get_object_coordinate(result_vec, current_fps_det, current_fps_cap);
                        //show_console_result(result_vec, obj_names, detection_data.frame_id);
                        //large_preview.draw(draw_frame);
                        //small_preview.draw(draw_frame, true);

                        detection_data.result_vec = result_vec;
                        detection_data.draw_frame = draw_frame;
                        draw2show.send(detection_data);
                        if (send_network) draw2net.send(detection_data);
                        if (output_video.isOpened()) draw2write.send(detection_data);
                

			// Tambahan
			// Send Data with UDP LAN or localhost
			sprintf(str,"%d,%d,%d,%d,%d,%d,%d,%d,%d", ball_x, ball_y, (int)ball_d, (int)goal_d[1][0], (int)goal_d[1][1], (int)Lcross_d[1][0], (int)Lcross_d[1][1], (int)Xcross_d[1][0], (int)Xcross_d[1][1]);
			//sprintf(str,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", ball_x, ball_y, (int)ball_d, gX_under, gY_under, (int)gD_under, (int)pinalty_d, (int)Lcross_d[1][0], (int)Lcross_d[1][1], (int)Xcross_d[1][0], (int)Xcross_d[1][1], (int)Tcross_d[1][0], (int)Tcross_d[1][1]);
			//sendto(viSoc, str, BUFLEN, 0, &addrVision, sizeof(addrVision));
			if (sendto(lm, str , BUFLEN , 0 , (struct sockaddr *) &Remote, sizeof(Remote))==-1)
				die("sendto()");
                    } while (!detection_data.exit_flag);
                    std::cout << " t_draw exit \n";
                });

                // write frame to videofile
                t_write = std::thread([&]()
                {
                    if (output_video.isOpened()) {
                        detection_data_t detection_data;
                        cv::Mat output_frame;
                        do {
                            detection_data = draw2write.receive();
                            if(detection_data.draw_frame.channels() == 4) cv::cvtColor(detection_data.draw_frame, output_frame, CV_RGBA2RGB);
                            else output_frame = detection_data.draw_frame;
                            output_video << output_frame;
                        } while (!detection_data.exit_flag);
                        output_video.release();
                    }
                    std::cout << " t_write exit \n";
                });

                // send detection to the network
                t_network = std::thread([&]()
                {
                    if (send_network) {
                        detection_data_t detection_data;
                        do {
                            detection_data = draw2net.receive();

                            detector.send_json_http(detection_data.result_vec, obj_names, detection_data.frame_id, filename);

                        } while (!detection_data.exit_flag);
                    }
                    std::cout << " t_network exit \n";
                });


                // show detection
                detection_data_t detection_data;
                do {

                    steady_end = std::chrono::steady_clock::now();
                    float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                    if (time_sec >= 1) {
                        current_fps_det = fps_det_counter.load() / time_sec;
                        current_fps_cap = fps_cap_counter.load() / time_sec;
                        steady_start = steady_end;
                        fps_det_counter = 0;
                        fps_cap_counter = 0;
                    }

                    detection_data = draw2show.receive();
                    cv::Mat draw_frame = detection_data.draw_frame;

                     //if (extrapolate_flag) {
                    //    cv::putText(draw_frame, "extrapolate", cv::Point2f(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
                    //}

                    if(use_GUI) cv::imshow("BarelangFC - Vision", draw_frame);
                    int key = cv::waitKey(3);    // 3 or 16ms
                    if (key == 'f') show_small_boxes = !show_small_boxes;
                    if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
                    //if (key == 'e') extrapolate_flag = !extrapolate_flag;
                    if (key == 27) { exit_flag = true;}

                    //std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap << std::endl;
                } while (!detection_data.exit_flag);
                std::cout << " show detection exit \n";

                cv::destroyWindow("BarelangFC - Vision");
                // wait for all threads
                if (t_cap.joinable()) t_cap.join();
                if (t_prepare.joinable()) t_prepare.join();
                if (t_detect.joinable()) t_detect.join();
                if (t_post.joinable()) t_post.join();
                if (t_draw.joinable()) t_draw.join();
                if (t_write.joinable()) t_write.join();
                if (t_network.joinable()) t_network.join();

                break;

            }
            else if (file_ext == "txt") {    // list of image files
                std::ifstream file(filename);
                if (!file.is_open()) std::cout << "File not found! \n";
                else
                    for (std::string line; file >> line;) {
                        std::cout << line << std::endl;
                        cv::Mat mat_img = cv::imread(line);
                        std::vector<bbox_t> result_vec = detector.detect(mat_img);
                        show_console_result(result_vec, obj_names);
                        //draw_boxes(mat_img, result_vec, obj_names);
                        //cv::imwrite("res_" + line, mat_img);
                    }

            }
            else {    // image file
                // to achive high performance for multiple images do these 2 lines in another thread
                cv::Mat mat_img = cv::imread(filename);
                auto det_image = detector.mat_to_image_resize(mat_img);

                auto start = std::chrono::steady_clock::now();
                std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, mat_img.size().width, mat_img.size().height);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> spent = end - start;
                std::cout << " Time: " << spent.count() << " sec \n";

                //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required
                draw_boxes(mat_img, result_vec, obj_names);
                cv::imshow("window name", mat_img);
                show_console_result(result_vec, obj_names);
                cv::waitKey(0);
            }
#else   // OPENCV
            //std::vector<bbox_t> result_vec = detector.detect(filename);

            auto img = detector.load_image(filename);
            std::vector<bbox_t> result_vec = detector.detect(img);
            detector.free_image(img);
            show_console_result(result_vec, obj_names);
#endif  // OPENCV
        }
        catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { std::cerr << "unknown exception \n"; getchar(); }
        filename.clear();
    }

    return 0;
}
