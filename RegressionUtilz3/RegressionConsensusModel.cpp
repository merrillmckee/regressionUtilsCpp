#include "RegressionConsensusModel.h"

PointF RegressionConsensusModel::GetPositiveCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate)
{
    if (points.size() <= model.MinimumPoints)
    {
        index = -1;
        pointsWithoutCandidate = vector<PointF>();
        return PointF();
    }

    auto maxRegressionError = -99999999.9f;
    index = 0;
    for (auto i = 0; i < points.size(); ++i)
    {
        auto point = points[i];
        bool pointOnPositiveSide;
        auto error = CalculateError(model, point, pointOnPositiveSide);

        if (pointOnPositiveSide)
        {
            if (error > maxRegressionError)
            {
                maxRegressionError = error;
                index = i;
            }
        }
    }

    pointsWithoutCandidate = vector<PointF>(points);
    pointsWithoutCandidate.erase(pointsWithoutCandidate.begin() + index);

    return points[index];
}

PointF RegressionConsensusModel::GetNegativeCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate)
{
    if (points.size() <= model.MinimumPoints)
    {
        index = -1;
        pointsWithoutCandidate = vector<PointF>();
        return PointF();
    }

    auto maxRegressionError = -99999999.9f;
    index = 0;
    for (auto i = 0; i < points.size(); ++i)
    {
        auto point = points[i];
        bool pointOnPositiveSide;
        auto error = CalculateError(model, point, pointOnPositiveSide);

        if (!pointOnPositiveSide)
        {
            if (error > maxRegressionError)
            {
                maxRegressionError = error;
                index = i;
            }
        }
    }

    pointsWithoutCandidate = vector<PointF>(points);
    pointsWithoutCandidate.erase(pointsWithoutCandidate.begin() + index);

    return points[index];
}


PointF RegressionConsensusModel::GetInfluenceCandidate(vector<PointF> points, RegressionModel& model, int& index, vector<PointF>& pointsWithoutCandidate)
{
    if (points.size() <= model.MinimumPoints)
    {
        index = -1;
        pointsWithoutCandidate = vector<PointF>();
        return PointF();
    }

    auto maximumInfluence = 0.0f;
    index = 0;
    for (auto i = 0; i < points.size(); ++i)
    {
        auto point = points[i];
        auto dx = point.X - model.bias.x;
        auto dy = point.Y - model.bias.y;

        float influence;
        if (influenceError == InfluenceError::L1)
        {
            influence = (float)abs(dx + dy);
        }
        else // L2
        {
            influence = (float)(dx * dx + dy * dy);
        }

        if (influence > maximumInfluence)
        {
            maximumInfluence = influence;
            index = i;
        }
    }

    pointsWithoutCandidate = vector<PointF>(points);
    pointsWithoutCandidate.erase(pointsWithoutCandidate.begin() + index);

    return points[index];
}

float RegressionConsensusModel::RemovePointAndCalculateError(vector<PointF> pointsWithoutCandidate, RegressionModel& modelWithoutCandidate)
{
    modelWithoutCandidate.CalculateModel(pointsWithoutCandidate);
    return modelWithoutCandidate.AverageRegressionError;
}

// Derived class will use the appropriate least squares regression to initialize the model/original
// Returns 0 on success, returns non-zero on failure
int RegressionConsensusModel::Calculate(vector<PointF> points, float sensitivity)
{
    if (points.size() < model->MinimumPoints)
    {
        // Exit with error
        return 1;
    }

    // Calculate the initial model.  Set the initial inliers and outliers (empty) lists.
    inliers = points;
    outliers = vector<PointF>();
    model->CalculateModel(points);
    original = model->Clone();

    // Keep removing candidate points until the model is lower than some average error threshold
    while (model->AverageRegressionError > sensitivity && model->ValidRegressionModel)
    {
        int index1, index2, index3;
        auto pointsWithoutPoint1 = vector<PointF>();
        auto pointsWithoutPoint2 = vector<PointF>();
        auto pointsWithoutPoint3 = vector<PointF>();
        auto candidatePoint1 = GetPositiveCandidate(inliers, *model, index1, pointsWithoutPoint1);
        auto candidatePoint2 = GetNegativeCandidate(inliers, *model, index2, pointsWithoutPoint2);
        auto candidatePoint3 = GetInfluenceCandidate(inliers, *model, index3, pointsWithoutPoint3);

        if (candidatePoint1.IsEmpty || candidatePoint2.IsEmpty || candidatePoint3.IsEmpty || index1 < 0 || index2 < 0 || index3 < 0)
        {
            // Exit with error
            break;
        }

        RegressionModel* modelWithoutPoint1 = model->Clone();
        RegressionModel* modelWithoutPoint2 = model->Clone();
        RegressionModel* modelWithoutPoint3 = model->Clone();
        auto newAverageError1 = RemovePointAndCalculateError(pointsWithoutPoint1, *modelWithoutPoint1);
        auto newAverageError2 = RemovePointAndCalculateError(pointsWithoutPoint2, *modelWithoutPoint2);
        auto newAverageError3 = RemovePointAndCalculateError(pointsWithoutPoint3, *modelWithoutPoint3);
        delete model;

        if (newAverageError1 < newAverageError2 && newAverageError1 < newAverageError3)
        {
            inliers = pointsWithoutPoint1;
            outliers.push_back(candidatePoint1);
            model = modelWithoutPoint1;
            delete modelWithoutPoint2; delete modelWithoutPoint3;
        }
        else if (newAverageError2 < newAverageError3)
        {
            inliers = pointsWithoutPoint2;
            outliers.push_back(candidatePoint2);
            model = modelWithoutPoint2;
            delete modelWithoutPoint1; delete modelWithoutPoint3;
        }
        else
        {
            inliers = pointsWithoutPoint3;
            outliers.push_back(candidatePoint3);
            model = modelWithoutPoint3;
            delete modelWithoutPoint1; delete modelWithoutPoint2;
        }
    }

    return 0;
}