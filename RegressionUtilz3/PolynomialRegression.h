#pragma once

#include "RegressionModel.h"

/// <summary>
/// PolynomialModel
/// Author: Merrill McKee
/// Description:  This is the abstract parent class for linear, quadratic, cubic, and 
///   any other polynomial regression algorithms. (todo: combine into single polynomial regression)
///   
/// </summary>
class PolynomialModel : public RegressionModel
{
protected:
    PolynomialModel()
    {
        _degree = DegreeOfPolynomial::Linear;
        independentVariable = enmIndependentVariable::X;
    }

    PolynomialModel(const PolynomialModel& copy)
        : RegressionModel(copy)
    {
        _degree = copy._degree;
        independentVariable = copy.independentVariable;
    }

public:

    enum class DegreeOfPolynomial
    {
        Linear = 1,
        Quadratic = 2,
        Cubic = 3
    };
    DegreeOfPolynomial _degree;

    enum class enmIndependentVariable                 // Which variable is independent?
    {                                           //   Linear, x-independent:     Can model horizontal lines
        X = 0,                                  //   Linear, y-independent:     Can model vertical lines
        Y                                       //   Quadratic, x-independent:  Vertical parabola
    };                                          //   Quadratic, y-independent:  Horizontal parabola
    enmIndependentVariable independentVariable;

    PolynomialModel& operator=(const PolynomialModel& other)
    {
        RegressionModel::operator=(other);

        _degree = other._degree;
        independentVariable = other.independentVariable;

        return *this;
    }

    // Returns the modeled y-value of a regression model with independent x-variable
    virtual float ModeledY(float x) = 0;

    // Returns the modeled x-value of a regression model with independent y-variable
    virtual float ModeledX(float y) = 0;

    // Calculate the single-point regression error
    float CalculateRegressionError(PointF point) override;

    // Return the degree of the regression model
    unsigned int Degree();
};