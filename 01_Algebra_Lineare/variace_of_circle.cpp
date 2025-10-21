#include <iostream>
#include <iomanip>
#include <random>

int main()
{
    int const N_ITERATIONS{1'000'000'000};
    std::default_random_engine engine{std::random_device{}()};
    std::uniform_real_distribution<double> distr{-1., 1.};

    double mean{0};
    double variance{0.};

    int N = 0;
    for (int i = 0; i != N_ITERATIONS; ++i)
    {
        double x{distr(engine)};
        double y{distr(engine)};
        if (x * x + y * y <= 1)
        {
            mean += x;
            variance += x * x;
            ++N;
        }
    }
    mean /= N;
    variance /= N;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Variance: " << variance << std::endl;
}