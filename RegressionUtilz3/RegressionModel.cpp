#include "RegressionModel.h"

const double RegressionModel::EPSILON = 0.0001;       // Near-zero value to check for division-by-zero

void RegressionModel::CalculateModel(vector<PointF> points)
{
    // Calculate the bias
    bias = CalculateBias(points);
    if (bias.x == 99999999.9)
    {
        ValidRegressionModel = false;
        return;
    }

    // Remove the bias
    auto pointsNoBias = RemoveBias(points, bias);
    if (pointsNoBias.size() == 0)
    {
        ValidRegressionModel = false;
        return;
    }

    // Calculate the summations on the points after the bias has been removed
    auto sum = CalculateSummations(pointsNoBias);
    if (sum->N <= 0)
    {
        ValidRegressionModel = false;
        return;
    }

    CalculateModel(*sum);
    if (!ValidRegressionModel)
    {
        return;
    }
    delete sum;

    CalculateFeatures();
    if (!ValidRegressionModel)
    {
        return;
    }

    CalculateAverageRegressionError(points);
    if (AverageRegressionError >= 99999999.9f)
    {
        ValidRegressionModel = false;
        return;
    }
}

RegressionModel::Bias RegressionModel::CalculateBias(vector<PointF> points)
{
    Bias bias;
    if (points.size() < 2)
    {
        // The minimum number of points to define an elliptical regression is 5
        bias.x = 9999999.9;
        bias.y = 9999999.9;
        return bias;
    }

    // Shorthand that better matches the math formulas
    auto N = points.size();

    //// Remove the bias (i.e. center the data at zero)
    //// Calculate the mean of a set of points
    auto meanX = 0.0;
    auto meanY = 0.0;
    for (auto i = 0; i < N; ++i)
    {
        meanX += points[i].X;
        meanY += points[i].Y;
    }
    meanX /= (float)N;
    meanY /= (float)N;
    bias.x = meanX;
    bias.y = meanY;

    return bias;
}

vector<PointF> RegressionModel::RemoveBias(vector<PointF> points, Bias bias)
{
    if (points.size() < 2)
    {
        return vector<PointF>();
    }

    if (bias.x == 9999999.9)
    {
        return vector<PointF>();
    }

    // Shorthand that better matches the math formulas
    auto N = points.size();

    //// Remove the mean from the set of points
    auto pointsNoBias = vector<PointF>();
    for (auto i = 0; i < N; ++i)
    {
        auto x = points[i].X - (float)bias.x;
        auto y = points[i].Y - (float)bias.y;
        pointsNoBias.push_back(PointF(x, y));
    }

    return pointsNoBias;
}

// Calculate the average regression error
float RegressionModel::CalculateAverageRegressionError(vector<PointF> points)
{
    if (points.size() == 0)
    {
        return 9999999.9f;
    }

    if (!ValidRegressionModel)
    {
        return 9999999.9f;
    }

    auto sumRegressionErrors = 0.0f;
    for (int i = 0; i < points.size(); ++i)
    {
        sumRegressionErrors += CalculateRegressionError(points[i]);
    }

    // Save internally
    AverageRegressionError = sumRegressionErrors / (float)points.size();

    // Also return
    return AverageRegressionError;
}

// If the bias is known or a good estimate exists, remove it
vector<PointF> RegressionModel::ZeroBiasPoints(vector<PointF> points, float xBias, float yBias)
{
    if (points.size() == 0)
    {
        // Invalid input
        return vector<PointF>();
    }

    auto newPoints = vector<PointF>();

    // Remove the bias
    for (auto i = 0; i < points.size(); ++i)
    {
        newPoints.push_back(PointF(points[i].X - xBias, points[i].Y - yBias));
    }

    return newPoints;
}

// In an attempt to remove unknown bias, zero mean a set of points
vector<PointF> RegressionModel::ZeroMeanPoints(vector<PointF> points, float & xMean, float & yMean)
{
    if (points.size() == 0)
    {
        // Invalid input
        xMean = 0.0f;
        yMean = 0.0f;
        return vector<PointF>();
    }

    auto xSum = 0.0f;
    auto ySum = 0.0f;
    auto newPoints = vector<PointF>();

    // Calculate the summations
    for (auto i = 0; i < points.size(); ++i)
    {
        xSum += points[i].X;
        ySum += points[i].Y;
    }
    xMean = xSum / (float)points.size();
    yMean = ySum / (float)points.size();

    // Zero the means
    for (auto i = 0; i < points.size(); ++i)
    {
        newPoints.push_back(PointF(points[i].X - xMean, points[i].Y - yMean));
    }

    return newPoints;
}