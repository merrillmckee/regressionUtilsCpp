#pragma once
#include <vector>

#include "PointF.cpp"
#include "RegressionModel.h"

using namespace std;

/// <summary>
/// Author: Merrill McKee
/// Description:  A set of inliers and outliers plus 2 regression models:
///   the original least squares regression model (all data points)
///   the consensus least squares regression model (only inliers)
/// </summary>
/// 
class RegressionConsensusModel
{
public:
    RegressionModel* model;
    RegressionModel* Model = model;
    RegressionModel* original;
    RegressionModel* Original = original;

    vector<PointF> inliers;
    vector<PointF>& Inliers = inliers;
    vector<PointF> outliers;
    vector<PointF>& Outliers = outliers;

    virtual RegressionConsensusModel& operator=(const RegressionConsensusModel& other)
    {
        model = other.model;
        original = other.original;
        inliers = other.inliers;
        outliers = other.outliers;

        return *this;
    }

protected:
    virtual float CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide) = 0;

    PointF GetPositiveCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate);
    PointF GetNegativeCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate);

    enum class InfluenceError
    {
        L1 = 1,
        L2 = 2,
    };
    InfluenceError influenceError = InfluenceError::L1;

    PointF GetInfluenceCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate);

    float RemovePointAndCalculateError(vector<PointF> pointsWithoutCandidate, RegressionModel& modelWithoutCandidate);

public:
    // Derived class will use the appropriate least squares regression to initialize the model/original
    // Returns 0 on success, returns non-zero on failure
    int Calculate(vector<PointF> points, float sensitivity);
};