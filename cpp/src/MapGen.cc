#include "MapGen.hpp"

MapGen::MapGen(uchar dismax, uchar grow_t = 5, uchar decay_t = 8, uchar period):
    disaster_max(dismax), _grow_t(grow_t), _decay_t(decay_t + grow_t), _period(period)
{
    exitMax = 0;
    dev_cnt = 0;
    abs_time = 0;
    map = cv::imread("../data/test.bmp", 0);

    exits = {cv::Point(3, 4), cv::Point(20, 8), cv::Point(27, 10), cv::Point(20, 13), cv::Point(3, 16)};
    outbreak = {cv::Point(11, 18), cv::Point(37, 10)};
}

void MapGen::display(const cv::Mat& src, cv::Mat& dst){
    dst.create(cv::Size(SQR_SZ * src.cols, SQR_SZ * src.rows), CV_8UC3);
    if (src.type() == CV_8UC1){
        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                uchar c = src.at<uchar>(i, j);
                cv::rectangle(dst, cv::Point(j * SQR_SZ, i * SQR_SZ),
                    cv::Point((j + 1) * SQR_SZ - 1, (i + 1) * SQR_SZ - 1), cv::Scalar(0, 50, 60), -1);
                cv::rectangle(dst, cv::Point(j * SQR_SZ + 1, i * SQR_SZ + 1),
                    cv::Point((j + 1) * SQR_SZ - 2, (i + 1) * SQR_SZ - 2), cv::Scalar(c, c, c), -1);
            }
        }
    }
    else{
        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                cv::Vec3b _vec = src.at<cv::Vec3b>(i, j);
                cv::Scalar color(_vec(0), _vec(1), _vec(2));
                cv::rectangle(dst, cv::Point(j * SQR_SZ, i * SQR_SZ),
                    cv::Point((j + 1) * SQR_SZ - 1, (i + 1) * SQR_SZ - 1), cv::Scalar(0, 50, 60), -1);
                cv::rectangle(dst, cv::Point(j * SQR_SZ + 1, i * SQR_SZ + 1),
                    cv::Point((j + 1) * SQR_SZ - 2, (i + 1) * SQR_SZ - 2), color, -1);
            }
        }
    }
}

template <calcType T>
uchar MapGen::calcSpreadScore(
    std::vector<cv::Point3i>& scores,
    const cv::Mat& score_map,
    const std::vector<cv::Point>& pts
) const{
    std::vector<std::vector<cv::Point3i> > potentials(pts.size());
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < pts.size(); i++){
        std::vector<cv::Point3i>& pot = potentials[i];
        const cv::Point& pt = pts[i];
        std::queue<cv::Point3i> st;
        Set searched;
        int dir_len;
        if (T == calcExit){
            dir_len = 4;
            st.emplace(pt.x, pt.y, 0);
            pot.emplace_back(pt.x, pt.y, 0);
        }
        else{
            dir_len = 8;
            st.emplace(pt.x, pt.y, disaster_max);
            pot.emplace_back(pt.x, pt.y, disaster_max);
        }
        searched.emplace(pt.x, pt.y);

        while (st.empty() == false){
            cv::Point3i cpos = st.front();
            st.pop();
            for (int i = 0; i < dir_len; i++){
                cv::Point3i mv_pos = (T == calcExit) ? cpos + dirs[i] : cpos - dirs[i];
                if  (mv_pos.x >= 0 && mv_pos.x < score_map.cols &&
                    mv_pos.y >= 0 && mv_pos.y < score_map.rows && mv_pos.z > 0
                ){
                    if (searched.find(std::make_pair(mv_pos.x, mv_pos.y)) == searched.end() &&
                        score_map.at<uchar>(mv_pos.y, mv_pos.x) == 0
                    ){
                        searched.emplace(mv_pos.x, mv_pos.y);
                        st.emplace(mv_pos);
                        pot.emplace_back(mv_pos.x, mv_pos.y, std::min(mv_pos.z, 255));
                    }
                }
            }
        }
        printf("Score for exit NO.%lu calculated.\n", i);
    }
    Dict container;
    for (const cv::Point3i& pt: potentials.front()){
        container[std::make_pair(pt.x, pt.y)] = pt.z;
    }
    for (size_t k = 1; k < potentials.size(); k++){
        const std::vector<cv::Point3i>& pot_ref = potentials[k];
        #pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < pot_ref.size(); i++){
            const cv::Point3i& pt = pot_ref[i];
            std::pair<int, int> pr(pt.x, pt.y);
            Dict::iterator it = container.find(pr);
            if (it != container.end()){
                if (T == calcExit){                     // exit score selects the minima
                    it->second = std::min(pt.z, it->second);
                }
                else {                                  // outbreakscore uses summation
                    it->second += pt.z;
                }
            }
            else{
                container[pr] = pt.z;
            }
        }
    }
    uchar maxVal = 0;
    for (Dict::const_iterator cit = container.begin(); cit != container.end(); cit++){
        uchar tmp = cit->second;
        if (T ==  calcExit && tmp > maxVal){
            maxVal = tmp;
        }
        scores.emplace_back(cit->first.first, cit->first.second, tmp);
    }
    return maxVal;
}

void MapGen::calcScoresDraw(cv::Mat& score){
    cv::Mat score_map, to_save;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cv::threshold(map, score_map, 254, 255, cv::THRESH_BINARY);
        }
        #pragma omp section
        {
            cv::threshold(map, to_save, 254, 255, cv::THRESH_BINARY);
        }
    }
    if (abs_time == 0){
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                exitMax = calcSpreadScore<calcExit>(exit_score, score_map, exits);
                printf("Maximum point counter in exit score(%lu): %d\n", exit_score.size(), exitMax);
            }
            #pragma omp section
            {
                calcSpreadScore<calcDist>(dist_score, score_map, outbreak);
                printf("Maximum point counter in dist score(%lu): %d\n", dist_score.size(), disaster_max);
            }
        }
    }
    else{
        timeDevelops(to_save);
    }
    prepDrawing(score_map, score);
    abs_time ++;
    if (abs_time >= _grow_t){
        std::cout << "Now it is decaying.\n";
        dev_cnt --;
    }
    else{
        std::cout << "Now it is growing.\n";
        dev_cnt ++;
    }
}

void MapGen::prepDrawing(cv::Mat& src, cv::Mat& dst) const{
    src.forEach<uchar>(
        [&](uchar &pix, const int* pos) -> void{
            if (pix != 0xff)
                pix = 0;
        }
    );
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < exit_score.size(); i++){
        const cv::Point3i& pt = exit_score[i];
        src.at<uchar>(pt.y, pt.x) = pt.z;
    }
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < dist_score.size(); i++){
        const cv::Point3i& pt = dist_score[i];
        uchar& pt_ref = src.at<uchar>(pt.y, pt.x);
        src.at<uchar>(pt.y, pt.x) += pt.z;
    }
    saveAsBinary(src, "../data/test.bin");
    convertTo(src, dst, disaster_max + exitMax, COLOR_R);
}

void MapGen::timeDevelops(const cv::Mat& occ){
    if (abs_time < _grow_t){        // disaster develops
        std::queue<cv::Point3i> pts;
        Set searched;
        for (cv::Point3i& pt: dist_score){
            if (pt.z == 1){
                pts.emplace(pt.x, pt.y, 2);
                searched.emplace(pt.x, pt.y);
            }
            pt.z ++;
        }
        while(pts.empty() == false){
            cv::Point3i cpos = pts.front();
            pts.pop();
            for (int i = 0; i < 4; i++){
                cv::Point3i mv_pos = cpos - dirs[i];
                if  (mv_pos.x >= 0 && mv_pos.x < occ.cols &&
                    mv_pos.y >= 0 && mv_pos.y < occ.rows
                ){
                    Set::iterator cit = searched.find(std::make_pair(mv_pos.x, mv_pos.y));
                    if (cit == searched.end() && occ.at<uchar>(mv_pos.y, mv_pos.x) == 0){
                        searched.emplace(mv_pos.x, mv_pos.y);
                        dist_score.emplace_back(mv_pos.x, mv_pos.y, 1);
                    }
                }
            }
        }
    }
    else{           // disaster decays
        #pragma omp parallel for num_threads(8)
        for (size_t i = 0; i < dist_score.size(); i++){
            dist_score[i].z --;
        }
        for (std::vector<cv::Point3i>::iterator it = dist_score.begin(); it != dist_score.end();){
            if (it->z == 0){
                it = dist_score.erase(it);
            }
            else{
                ++ it;
            }
        }
    }
}

void MapGen::convertTo(const cv::Mat& src, cv::Mat& dst, uchar norm_val, COLOR color){
    dst.create(src.size(), CV_8UC3);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            uchar val = src.at<uchar>(i, j);
            if (val < 0xff){
                val = std::min(double(val) / double(norm_val) * 255, (double)255);
                if (color == COLOR_R)
                    dst.at<cv::Vec3b>(i, j) = cv::Vec3b(!val, 0, val);
                else if (color == COLOR_G)
                    dst.at<cv::Vec3b>(i, j) = cv::Vec3b(0, val, 0);
                else
                    dst.at<cv::Vec3b>(i, j) = cv::Vec3b(val, 0, !val);
            }
            else{
                dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            }
        }
    }
}

void MapGen::saveAsBinary(const cv::Mat& scores, std::string opath){
    std::ofstream file;
    file.open(opath, std::ios::out | std::ios::binary);
    const uchar* data = scores.data;
    file.write((char *)data, scores.cols * scores.rows * sizeof(uchar));
    file.close();
}

void onMouseEvent(int event, int x, int y, int flags, void* userdata){
    if (event == cv::EVENT_LBUTTONDOWN){
        printf("Click on img(%d, %d)\n", x, y);
        int posx = x / SQR_SZ, posy = y / SQR_SZ;
        printf("Click on square(%d, %d)\n", posx, posy);
    }
}