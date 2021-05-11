#pragma once
#include <vector>

#include "PointF.cpp"

using namespace std;

/// <summary>
/// RegressionModel
/// Author: Merrill McKee
/// Description:  This is the abstract parent class for regression models PolynomialModel (abstract) and EllipseModel
///   
/// </summary>
/// 
class RegressionModel
{
protected:
    const static double EPSILON;       // Near-zero value to check for division-by-zero
    
public:

    RegressionModel()
    {
        ValidRegressionModel = false;
        averageRegressionError = 99999999.9f;
        bias.x = 0;
        bias.y = 0;
        MinimumPoints = 2;
    }

public:

    RegressionModel(const RegressionModel& copy)
    {
        ValidRegressionModel = copy.ValidRegressionModel;
        averageRegressionError = copy.AverageRegressionError;
        bias = copy.bias;
        MinimumPoints = copy.MinimumPoints;
    }

    virtual RegressionModel* Clone() = 0;

    virtual RegressionModel& operator=(const RegressionModel& other)
    {
        ValidRegressionModel = other.ValidRegressionModel;
        averageRegressionError = other.AverageRegressionError;
        bias = other.bias;
        MinimumPoints = other.MinimumPoints;

        return *this;
    }

    int MinimumPoints;
    bool ValidRegressionModel;        // Remains false until a model is successfully created

    float averageRegressionError;
    float& AverageRegressionError = averageRegressionError;

    struct Bias
    {
        double x;
        double y;
    };
    Bias bias;

    virtual void CalculateFeatures() = 0;

    class Summations
    {
    public:
        int N;
        double x;   // Sigma(x)
        double y;   // Sigma(y)

        Summations()
        {
            N = 0;
            x = 0;
            y = 0;
        }

        Summations(const Summations& copy)
        {
            N = copy.N;
            x = copy.x;
            y = copy.y;
        }
    };

    virtual Summations* CalculateSummations(vector<PointF> points) = 0;

    virtual void CalculateModel(Summations& sum) = 0;

    void CalculateModel(vector<PointF> points);

    static Bias CalculateBias(vector<PointF> points);
    static vector<PointF> RemoveBias(vector<PointF> points, Bias bias);
    float CalculateAverageRegressionError(vector<PointF> points);

    // Calculate the single-point regression error
    virtual float CalculateRegressionError(PointF point) = 0;

    // If the bias is known or a good estimate exists, remove it
    static vector<PointF> ZeroBiasPoints(vector<PointF> points, float xBias, float yBias);

    // In an attempt to remove unknown bias, zero mean a set of points
    static vector<PointF> ZeroMeanPoints(vector<PointF> points, float& xMean, float& yMean);
};
