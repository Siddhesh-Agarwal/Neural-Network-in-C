#include <stdio.h>

/// @brief Linear regression prediction
/// @param x - x values
/// @param y - y values
/// @param X - x value to predict
/// @param n - number of values
/// @return y value
double linearPredict(double x[], double y[], double X, int n)
{
    int sigma_x = 0, sigma_y = 0, sigma_x2 = 0, sigma_xy = 0;
    for (int i = 0; i < n; i++)
    {
        sigma_x += x[i];
        sigma_y += y[i];
        sigma_x2 += x[i] * x[i];
        sigma_xy += x[i] * y[i];
    }

    double denominator = n * sigma_x2 - sigma_x * sigma_x;
    double a = (sigma_y * sigma_x2 - sigma_x * sigma_xy) / denominator;
    double b = (sigma_xy * n - sigma_x * sigma_y) / denominator;

    return a + b * X;
}

/// @brief Polynomial regression prediction
/// @param x - x values
/// @param y - y values
/// @param X - x value to predict
/// @param h - degree of polynomial
/// @param n - number of values
double polyPredict(double x[], double y[], double X, int h, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        double product = 1;
        for (int j = 0; j < h; j++)
        {
            product *= (X - x[i]);
        }
        sum += y[i] * product;
    }
    return sum;
}