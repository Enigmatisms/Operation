#include <iostream>
#include <chrono>
#include "MapGen.hpp"

int main(int argc, char* argv[]){
    uchar dismax = 12, gtime = 12, dtime = 12, period = 1;
    if (argc < 5){
        std::cout << "Usage: ./Task -max_disaster_impact -disaster_grow_time -disaster_decay_time (-period)\n";
    }
    if (argc > 1) dismax    = (uchar)atoi(argv[1]);
    if (argc > 2) gtime     = (uchar)atoi(argv[2]);
    if (argc > 3) dtime     = (uchar)atoi(argv[3]);
    if (argc > 4) period    = (uchar)atoi(argv[4]);

    MapGen mpg(dismax, gtime, dtime, period);
    cv::Mat score, to_show;
    std::chrono::system_clock clk;
    
    cv::namedWindow("disp");
    cv::setMouseCallback("disp", onMouseEvent);
    while(true){
        uint64_t start_time = clk.now().time_since_epoch().count();
        mpg.calcScoresDraw(score);
        uint64_t end_time = clk.now().time_since_epoch().count();
        printf("Calculation takes %ld nano secs, which is %f ms.\n", end_time - start_time, (float)((end_time - start_time) / 1e6));
        mpg.display(score, to_show);
        cv::imshow("disp", to_show);
        char key = cv::waitKey(0);
        if (key == 27){
            break;
        }
    }
    
    return 0;
}