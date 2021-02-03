#ifndef __MAP_GEN_HPP
#define __MAP_GEN_HPP

/**
 *   Lourve Map Generator and preprocessor.
*/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <queue>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#define SQR_SZ 24

const cv::Point3i dirs[8] = {
    cv::Point3i(-1, 0, 1),
    cv::Point3i(1, 0, 1),
    cv::Point3i(0, -1, 1),
    cv::Point3i(0, 1, 1),
    cv::Point3i(1, 1, 1),
    cv::Point3i(1, -1, 1),
    cv::Point3i(-1, 1, 1),
    cv::Point3i(-1, -1, 1),
};

enum COLOR{             // color to render
    COLOR_B = 0,
    COLOR_G = 1,
    COLOR_R = 2
};

enum calcType{
    calcExit = 0,
    calcDist = 1
};

struct hashFunctor{
    template<typename T>
    size_t operator() (const std::pair<T, T>& pt) const{
        auto h1 = std::hash<T>{}(pt.first);
        auto h2 = std::hash<T>{}(pt.second);
        return h1 ^ h2;
    }
};

struct equalFunctor{
    template<typename T>
    bool operator() (const std::pair<T, T>& p1, const std::pair<T, T>& p2) const{
        return (p1.first == p2.first) && (p1.second == p2.second);
    }
};

void onMouseEvent(int event, int x, int y, int flags, void* userdata);

class MapGen{
using Dict = std::unordered_map<std::pair<int, int>, int, hashFunctor, equalFunctor>;
using Set = std::unordered_set<std::pair<int, int>, hashFunctor, equalFunctor>;
public:
    MapGen(uchar dismax, uchar grow_t, uchar decay_t, uchar period = 1);
    ~MapGen(){;}
public:
    /**
     * @brief calculate the distance between the current pos to the nearest exit.
     */
    void calcScoresDraw(cv::Mat& score);

    static void display(const cv::Mat& src, cv::Mat& dst);
private:
    /**
     * @brief how disasters develops as time goes by: start >>> grow >>> peak >>> decay >>> none;
     * 
     */
    void timeDevelops(const cv::Mat& occ);

    void prepDrawing(cv::Mat& src, cv::Mat& dst) const;

    /**
     * @brief calculate potential using queue and dict structure
     * @arg template calcType: indicate whether we are calculating the scores of exits or outbreak points
     * @param scores score container
     * @param score_map cv::Mat indicates the occupancy
     * @param pts which point list to start from
     * @return max counter
     */
    template<calcType T>
    uchar calcSpreadScore(
        std::vector<cv::Point3i>& scores,
        const cv::Mat& score_map,
        const std::vector<cv::Point>& pts
    ) const;

    static void convertTo(const cv::Mat& src, cv::Mat& dst, uchar norm_val, COLOR color);

    static void saveAsBinary(const cv::Mat& scores, std::string opath);
public:
    cv::Mat map;
    cv::RNG rng;
    std::vector<cv::Point> outbreak;        // outbreak points
    std::vector<cv::Point> exits;           // exits
    std::vector<cv::Point3i> dist_score;    // save to reuse
    std::vector<cv::Point3i> exit_score;    // save to reuse
    uchar disaster_max, exitMax;
    uchar dev_cnt;                          // counter recording the development of outbreak
    uchar abs_time;                         // absolute time coordinate
    const uchar _grow_t;
    const uchar _decay_t;
    const uchar _period;                    // disaster develops every _period (times of iteration)
};

#endif  //__MAP_GEN_HPP