#include <iostream>
#include <vector>
#include <cmath>

struct Stats{
    double mean;
    double stddev;
    double sum;
    double sumSq;
    int n;
};


Stats computeStats(const std::vector<int>& v)
{
    // calculate mean
    double mean = 0.0;
    for(int x : v) mean += x;
    mean /= v.size();

    // calculate variance
    double variance = 0.0;
    for(int x : v){
        variance += (x - mean) * (x - mean);
    }
    variance /= v.size();

    // calcualte sum
    double sum = 0.0;
    double sumSq = 0.0;

    for(int x : v) {
        sum += x;
        sumSq += sum * sum;
    }

    return { mean , std::sqrt(variance), sum, sumSq, (int)v.size() };
}



int main()
{
    std::vector<std::vector<int>> data = { {99, 2, 3, 3, 4, 5}, {1,3,4,5,5,2} };
    
    for (const auto& row : data) {
        Stats s = computeStats(row);
        std::cout << s.mean << " " << s.stddev << " " << s.sum << " " << s.sumSq << " " << s.n <<"\n";
    }

    return 0;
}