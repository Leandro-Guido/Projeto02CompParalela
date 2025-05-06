#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <sstream>

using namespace std;

/*
Kmeans
Adaptado de https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
*/

struct Point {
    double x, y, z;
    int cluster;
    Point(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z), cluster(-1) {}
};

double euclidean_distance(const Point& a, const Point& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

void kmeans(vector<Point>& points, int epochs = 100, int k = 3) {
    int n = points.size();
    vector<Point> centroids;
    srand(time(nullptr));

    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[rand() % n]);
    }

    for (int e = 0; e < epochs; ++e) {
        // Atribuir pontos ao cluster mais próximo
        for (auto& p : points) {
            double min_dist = numeric_limits<double>::max();
            int closest = -1;
            for (int cluster = 0; cluster < k; ++cluster) {
                double dist = euclidean_distance(p, centroids[cluster]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = cluster;
                }
            }
            p.cluster = closest;
        }
        
        vector<Point> old_centroids = centroids;

        // Atualizar centroides
        vector<int> count(k, 0);
        vector<double> sum_x(k, 0), sum_y(k, 0), sum_z(k, 0);

        for (const auto& p : points) {
            int cluster = p.cluster;
            sum_x[cluster] += p.x;
            sum_y[cluster] += p.y;
            sum_z[cluster] += p.z;
            count[cluster]++;
        }

        for (int cluster = 0; cluster < k; ++cluster) {
            if (count[cluster] > 0) {
                centroids[cluster].x = sum_x[cluster] / count[cluster];
                centroids[cluster].y = sum_y[cluster] / count[cluster];
                centroids[cluster].z = sum_z[cluster] / count[cluster];
            }
        }
    }
}

/*
Utils e processamento de dados
*/

vector<Point> readcsv(const string& filename) {
    ifstream file(filename);
    vector<Point> points;
    string line;

    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        return points;
    }

    getline(file, line); // cabecalho

    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        getline(ss, token, ',');
        double x = stod(token);
        getline(ss, token, ',');
        double y = stod(token);
        getline(ss, token, ',');
        double z = stod(token);

        points.emplace_back(x, y, z);
    }

    return points;
}

int main() {
    vector<Point> pontos = readcsv("housing_pre.csv");

    int k = 3;
    int epochs = 10000;
    kmeans(pontos, epochs, k);

    // Salvar resultado
    ofstream out("output.csv");
    out << "longitude,latitude,median_house_value,cluster\n";
    for (const auto& p : pontos) {
        out << p.x << "," << p.y << "," << p.z << "," << p.cluster << "\n";
    }

    cout << "K-means finalizado. Resultado salvo em output.csv\n";
    return 0;
}
