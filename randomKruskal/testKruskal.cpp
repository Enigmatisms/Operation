#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
extern "C"{
#include "./include/kruskal.h"
}

int main(){
    initialize();
    breakWalls();
    printDebug();
    while (true){
        cv::Mat src(320, 640, CV_8UC1);
        for (int i = 0; i < 32; i++){
            for (int j = 0; j < 16; j++){
                if (backGround[i + 32 * j] == 0){
                    cv::rectangle(src, cv::Rect(i * 20, j * 20, 20, 20), cv::Scalar(0), -1);
                }
                else{
                    if (backGround[i + 32 * j] == 1){
                        cv::rectangle(src, cv::Rect(i * 20, j * 20, 20, 20), cv::Scalar(255), -1);
                    }
                    else if (backGround[i + 32 * j] == 128){
                        cv::rectangle(src, cv::Rect(i * 20, j * 20, 20, 20), cv::Scalar(150), -1);
                    }
                    else{
                        cv::rectangle(src, cv::Rect(i * 20, j * 20, 20, 20), cv::Scalar(100), -1);
                    }
                }
                cv::rectangle(src, cv::Rect(i * 20, j * 20, 20, 20), cv::Scalar(100));
            }
        }
        cv::imshow("kruscal_full", src);
        char key = cv::waitKey(0);
        if (key == 27){
            break;
        }
        else if (key == ' '){
            initialize();
            breakWalls();
            printDebug();
        }
        else if (key == 'w'){
            move(3);
        }
        else if (key == 'a'){
            move(1);
        }
        else if (key == 's'){
            move(2);
        }
        else if (key == 'd'){
            move(0);
        }
    }
    cv::destroyAllWindows();
    return 0;
}
