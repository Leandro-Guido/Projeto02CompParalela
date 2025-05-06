#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cfloat>

using namespace std;

struct Point {
    double x, y;
    int cluster;
    double minDist;

    Point() : x(0.0), y(0.0), cluster(-1), minDist(DBL_MAX) {}
    Point(double x, double y) : x(x), y(y), cluster(-1), minDist(DBL_MAX) {}

    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

// Utilitário para converter string de data "dd-mm-yyyy HH:MM" para time_t
time_t parseDate(const string& dateStr) {
    struct tm tm = {};
    strptime(dateStr.c_str(), "%d-%m-%Y %H:%M", &tm);
    return mktime(&tm);
}

vector<Point> readRFMFromCSV(string filename) {
    ifstream file(filename);
    string line;

    map<string, int> freq;
    map<string, double> monetary;
    map<string, time_t> lastPurchase;

    getline(file, line); // skip header

    time_t globalMaxTime = 0;

    while (getline(file, line)) {
        stringstream ss(line);
        string invoice, stock, desc, quantityStr, dateStr, priceStr, custID, country;

        getline(ss, invoice, ',');
        getline(ss, stock, ',');
        getline(ss, desc, ',');
        getline(ss, quantityStr, ',');
        getline(ss, dateStr, ',');
        getline(ss, priceStr, ',');
        getline(ss, custID, ',');
        getline(ss, country, '\n');

        if (custID.empty()) continue;

        try {
            if (quantityStr.empty() || priceStr.empty()) continue;
        
            int quantity = stoi(quantityStr);
            double unitPrice = stod(priceStr);
            time_t invoiceTime = parseDate(dateStr);
        
            // Atualiza valores agregados por cliente
            freq[custID] += 1;
            monetary[custID] += quantity * unitPrice;
        
            if (lastPurchase.find(custID) == lastPurchase.end() || invoiceTime > lastPurchase[custID]) {
                lastPurchase[custID] = invoiceTime;
            }
            if (invoiceTime > globalMaxTime) globalMaxTime = invoiceTime;

        } catch (const std::invalid_argument& e) {
            continue; // Pula a linha com erro
        } catch (const std::out_of_range& e) {
            continue; // Também ignora valores fora do intervalo
        }        
    }

    vector<Point> points;
    for (auto& [custID, count] : freq) {
        double recency = difftime(globalMaxTime, lastPurchase[custID]) / (60 * 60 * 24); // dias
        double frequency = count;
        double money = monetary[custID];

        // Use frequência e valor monetário (ou recência se quiser mudar)
        points.push_back(Point(frequency, money));
    }

    return points;
}

void kMeansClustering(vector<Point>* points, int epochs, int k) {
    int n = points->size();
    vector<Point> centroids;
    srand(time(0));

    for (int i = 0; i < k; ++i) {
        centroids.push_back(points->at(rand() % n));
    }

    for (int e = 0; e < epochs; ++e) {
        for (auto& c : centroids) {
            int clusterId = &c - &centroids[0];

            for (auto& p : *points) {
                double dist = c.distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
            }
        }

        vector<int> nPoints(k, 0);
        vector<double> sumX(k, 0.0), sumY(k, 0.0);

        for (auto& p : *points) {
            int clusterId = p.cluster;
            nPoints[clusterId]++;
            sumX[clusterId] += p.x;
            sumY[clusterId] += p.y;
            p.minDist = DBL_MAX;
        }

        for (int i = 0; i < k; ++i) {
            if (nPoints[i] > 0) {
                centroids[i].x = sumX[i] / nPoints[i];
                centroids[i].y = sumY[i] / nPoints[i];
            }
        }
    }

    ofstream out("output.csv");
    out << "Frequency,Monetary,Cluster\n";
    for (auto& p : *points) {
        out << p.x << "," << p.y << "," << p.cluster << "\n";
    }
    out.close();
}

int main() {
    vector<Point> points = readRFMFromCSV("OnlineRetail.csv");
    int k = 5;
    int epochs = 100;
    kMeansClustering(&points, epochs, k);
    cout << "K-means clustering complete. Results written to output.csv.\n";
    return 0;
}
