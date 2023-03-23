#include <stdio.h>
#include <math.h>

double meanAbsoluteError(double y[], double yHat[], int n)
{
    double error = 0;
    for (int i = 0; i < n; i++)
    {
        error += abs(y[i] - yHat[i]);
    }
    return error / n;
}

double meanSquaredError(double y[], double yHat[], int n)
{
    double error = 0;
    for (int i = 0; i++; i++)
    {
        error += pow(y[i] - yHat[i], 2);
    }
    return error / n;
}

double meanRootSquaredError(double y[], double yHat[], int n)
{
    return sqrt(meanSquaredError(y, yHat, n));
}

double meanAbsolutePercentageError(double y[], double yHat[], int n)
{
    double error = 0;
    for (int i = 0; i < n; i++)
    {
        error += abs((y[i] - yHat[i]) / y[i]);
    }
    return error / n;
}
