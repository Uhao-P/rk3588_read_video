#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>


#define CAM_DEVICE "/dev/video0"  // 摄像头设备文件路径
#define IMG_WIDTH 640  // 图像宽度
#define IMG_HEIGHT 480  // 图像高度

double what_time_is_it_now()
{
	// 单位: ms
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * .001;
}


int main() {
    // 打开摄像头设备
    int fd = open(CAM_DEVICE, O_RDWR);
    if (fd == -1) {
        perror("Error: failed to open camera.");
        return -1;
    }

    // 设置视频格式
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = IMG_WIDTH;
    fmt.fmt.pix.height = IMG_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Error: failed to set video format.");
        return -1;
    }

    // 申请缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Error: failed to request buffer.");
        return -1;
    }

    // 映射缓冲区
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Error: failed to query buffer.");
        return -1;
    }
    void *buf_start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buf_start == MAP_FAILED) {
        perror("Error: failed to mmap buffer.");
        return -1;
    }

    // 开始捕获图像
    if (ioctl(fd, VIDIOC_STREAMON, &buf.type) == -1) {
        perror("Error: failed to start stream.");
        return -1;
    }

    // // 创建OpenCV窗口
    // cv::namedWindow("camera", cv::WINDOW_AUTOSIZE);

    double end;
    double start;
    double fps;
    int count = 0;
    double sum = 0;
    // 读取摄像头数据并处理
    while (true) {
        printf("---------------------------\n");
        start = what_time_is_it_now();
        // 读取图像数据
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("Error: failed to queue buffer.");
            return -1;
        }
        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("Error: failed to dequeue buffer.");
            return -1;
        }

        // 将YUYV格式转换为BGR格式
        cv::Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC2, buf_start);
        cv::cvtColor(img, img, cv::COLOR_YUV2BGR_YUYV);
        // cv::imwrite("test.jpg", img);

        end = what_time_is_it_now();
        double t_diff = end - start;
        sum += t_diff;
        fps = 1000.0 / t_diff;
        count++;
        std::cout << "Frame " << count << ", time: " << t_diff << " s, FPS: " << fps << std::endl;

        // if (cv::waitKey(30) == 27) {
        //     break;
        // }
        // cv::imshow("camera", img);
    }
    double average_time = sum / count;
    std::cout << "Average time per frame: " << average_time << " s" << std::endl;

    // 释放缓冲区
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        perror("Error: failed to queue buffer.");
        return -1;
    }

    // 停止捕获图像并释放资源
    if (ioctl(fd, VIDIOC_STREAMOFF, &buf.type) == -1) {
        perror("Error: failed to stop stream.");
        return -1;
    }
    munmap(buf_start, buf.length);
    close(fd);

    return 0;
}
