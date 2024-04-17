#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rga.h"

using namespace std;
using namespace cv;

/*
*  测试librga与opencv的resize性能
*/


double what_time_is_it_now()
{
	// 单位: ms
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * .001;
}

int main()
{
    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    // opencv read img
    string img_path = "/udisk/workspaces/rknpu2/examples/rknn_yolov5_demo/model/bus.jpg";
    cv::Mat orig_img = cv::imread(img_path);
    int img_width  = orig_img.cols;
    int img_height = orig_img.rows;
    printf("img width = %d, img height = %d\n", img_width, img_height);

    // rga resize img
    int dst_width = 1920;
    int dst_height = 1080;
    void* resize_buf = nullptr;
    resize_buf = malloc(dst_height * dst_width * 3);
    memset(resize_buf, 0x00, dst_height * dst_width * 3);
    double rga_sum = 0;
    double rga_end;
    double rga_start;
    for (int i = 0; i < 30; i++) {
        
        src = wrapbuffer_virtualaddr((void*)orig_img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void*)resize_buf, dst_width, dst_height, RK_FORMAT_RGB_888);
        // int ret = imcheck(src, dst, src_rect, dst_rect);
        int ret = imcheck(src, dst, {}, {});
        rga_start = what_time_is_it_now();
        IM_STATUS STATUS = imresize(src, dst);
        rga_end = what_time_is_it_now();
        rga_sum += rga_end - rga_start;
    }
    double rga_one = rga_sum / 30;

    // val result img
    cv::Mat resize_img(cv::Size(dst_width, dst_height), CV_8UC3, resize_buf);
    cv::imwrite("rga_resize_val.jpg", resize_img);



    Mat resized_image;
    double opencv_sum = 0;
    double opencv_start;
    double opencv_end;
    for (int i = 0; i < 30; i++) {
        opencv_start = what_time_is_it_now();
        cv::resize(orig_img, resized_image, cv::Size(dst_width, dst_height));
        opencv_end = what_time_is_it_now();
        opencv_sum += opencv_end - opencv_start;
    }
    double opencv_one = opencv_sum / 30;
    cv::imwrite("opencv_resize_val.jpg", resized_image);

    std::cout << "rga cost time : " << rga_one << std::endl;
    std::cout << "opencv cost time : " << opencv_one << std::endl;
    std::cout << "rga ave cost time : " << rga_end - rga_start << std::endl;
    std::cout << "opencv ave cost time : " << opencv_end - opencv_start << std::endl;

    orig_img.release();
    resized_image.release();
    release_buffer:
        free(resize_buf);

    return 0;
}
