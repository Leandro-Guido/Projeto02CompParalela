#include <ctime>     // for a random seed
#include <fstream>   // for file-reading
#include <iostream>  // for file-reading
#include <sstream>   // for file-reading
#include <vector>
#include <cfloat>   // for __DBL_MAX__

using namespace std;

struct Point {
    double x, y;     // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite dist to nearest cluster

    Point() {
        this->x = 0.0;
        this->y = 0.0;
        this->cluster = -1;
        this->minDist = __DBL_MAX__;
    }

    Point(double x, double y) {
        this->x = x;
        this->y = y;
        this->cluster = -1;
        this->minDist = __DBL_MAX__;
    }

    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};