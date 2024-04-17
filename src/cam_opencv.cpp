#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
int main()
{   
    cv::VideoCapture cap;

    // 读摄像头的mjpeg数据
    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // cap.set(cv::CAP_PROP_CONVERT_RGB, false);

    // 读摄像头的yuv数据
    cap.set(cv::CAP_PROP_FORMAT, CV_8UC1);
    cap.set(cv::CAP_PROP_CONVERT_RGB, false);
    if (!cap.open(0)) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    double fps;
    double t = 0;
    double t_start = cv::getTickCount(); // 获取起始时间戳
    int count = 0;

    while (true) {
        cap >> frame; // 读取一帧图像
        if (frame.empty()) {
            break;
        }

        double t_end = cv::getTickCount(); // 获取结束时间戳
        double t_diff = (t_end - t_start) / cv::getTickFrequency(); // 计算时间差
        fps = 1.0 / t_diff; // 计算帧率

        t += t_diff;
        count++;

        std::cout << "Frame " << count << ", time: " << t_diff << " s, FPS: " << fps << std::endl;

        t_start = t_end;
    }

    double average_time = t / count;
    std::cout << "Average time per frame: " << average_time << " s" << std::endl;

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
