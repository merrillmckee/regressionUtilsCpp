#include <iostream>

#include "PolynomialRegression.h"

using namespace std;

// Calculate the single-point regression error
float PolynomialModel::CalculateRegressionError(PointF point)
{
    if (independentVariable == enmIndependentVariable::X)
    {
        return abs(ModeledY(point.X) - point.Y);
    }
    else
    {
        return abs(ModeledX(point.Y) - point.X);
    }
}

// Return the degree of the regression model
unsigned int PolynomialModel::Degree()
{
    return (unsigned int)_degree;
}