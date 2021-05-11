#include "CubicRegression.h"

const float CubicRegression::DEFAULT_SENSITIVITY = 0.35f;

float CubicRegression::CubicModel::ModeledY(float x)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::X)
    {
        return (float)(b1 + b2 * x + b3 * x * x + b4 * x * x * x);
    }
    else
    {
        return -99999999.9f;
    }
}

float CubicRegression::CubicModel::ModeledX(float y)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::Y)
    {
        return (float)(b1 + b2 * y + b3 * y * y + b4 * y * y * y);
    }
    else
    {
        return -99999999.9f;
    }
}

CubicRegression::CubicConsensusModel& CubicRegression::CalculateCubicRegressionConsensus(vector<PointF> points, enmIndependentVariable independentVariable, float sensitivity)
{
    auto consensus = new CubicConsensusModel(independentVariable);
    consensus->Calculate(points, sensitivity);

    return *consensus;
}

float CubicRegression::CubicConsensusModel::CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide)
{
    if (point.IsEmpty)
    {
        pointOnPositiveSide = false;
        return 99999999.9f;
    }

    auto error = 0.0f;
    auto qmodel = static_cast<CubicModel&>(model);
    if (qmodel.independentVariable == enmIndependentVariable::X)
    {
        error = qmodel.ModeledY(point.X) - point.Y;
    }
    else
    {
        error = qmodel.ModeledX(point.Y) - point.X;
    }

    pointOnPositiveSide = error >= 0.0f;
    return abs(error);
}

RegressionModel::Summations* CubicRegression::CubicModel::CalculateSummations(vector<PointF> points)
{
    CubicSummations* sum = new CubicSummations();
    if (points.size() < MinimumPoints)
    {
        sum->N = 0;
        return sum;
    }

    // Initialize all the summations to zero
    sum->x = 0.0;
    sum->y = 0.0;
    sum->xy = 0.0;
    sum->x2y = 0.0;  // (i.e.  SUM(x^2*y))
    sum->x3y = 0.0;
    sum->x2 = 0.0;
    sum->x3 = 0.0;
    sum->x4 = 0.0;
    sum->x5 = 0.0;
    sum->x6 = 0.0;

    // Shorthand that better matches the math formulas
    auto N = sum->N = (int)points.size();

    // Calculate the summations
    for (auto i = 0; i < N; ++i)
    {
        // Shorthand
        auto x = (double)points[i].X;
        auto y = (double)points[i].Y;

        // Meh
        if (independentVariable == enmIndependentVariable::Y)
        {
            // Swap the x and y coordinates to handle a y independent variable
            x = points[i].Y;
            y = points[i].X;
        }

        auto xx = x * x;
        auto xy = x * y;
        auto xxx = xx * x;

        // Sums
        sum->x += x;
        sum->y += y;
        sum->xy += xy;
        sum->x2y += xx * y;
        sum->x3y += xxx * y;
        sum->x2 += xx;
        sum->x3 += xxx;
        sum->x4 += xx * xx;
        sum->x5 += xxx * xx;
        sum->x6 += xxx * xxx;
    }

    return sum;
}

void CubicRegression::CubicModel::CalculateFeatures()
{

}

void CubicRegression::CubicModel::CalculateModel(Summations& sums)
{
    if (sums.N <= 0)
    {
        ValidRegressionModel = false;
        return;
    }

    CubicSummations& sum = static_cast<CubicSummations&>(sums);

    // Calculate the means
    auto XMean = sum.x / (double)sum.N;
    auto YMean = sum.y / (double)sum.N;
    auto XXMean = sum.x2 / (float)sum.N;
    auto XXXMean = sum.x3 / (double)sum.N;

    // Calculate the S intermediate values
    auto inv_N = (1.0 / (double)sum.N); // Shorthand
    auto s11 = sum.x2 - inv_N * sum.x * sum.x;
    auto s12 = sum.x3 - inv_N * sum.x * sum.x2;
    auto s13 = sum.x4 - inv_N * sum.x * sum.x3;
    auto s22 = sum.x4 - inv_N * sum.x2 * sum.x2;
    auto s23 = sum.x5 - inv_N * sum.x2 * sum.x3;
    auto s33 = sum.x6 - inv_N * sum.x3 * sum.x3;
    auto sY1 = sum.xy - inv_N * sum.x * sum.y;
    auto sY2 = sum.x2y - inv_N * sum.x2 * sum.y;
    auto sY3 = sum.x3y - inv_N * sum.x3 * sum.y;

    // Calculate the inverse matrix of S (inv(S)) using T notation
    // (see notes above)
    auto t11 = s22 * s33 - s23 * s23;
    auto t12 = s13 * s23 - s12 * s33;
    auto t13 = s12 * s23 - s13 * s22;
    auto t22 = s11 * s33 - s13 * s13;
    auto t23 = s12 * s13 - s11 * s23;
    auto t33 = s11 * s22 - s12 * s12;
    auto determinantS = s11 * (s22 * s33 - s23 * s23) - s12 * (s12 * s33 - s13 * s23) + s13 * (s12 * s23 - s13 * s22);

    // Don't divide by zero
    if (abs(determinantS) <= EPSILON)
    {
        ValidRegressionModel = false;
        return;
    }

    // Calculate the coefficients of y = b1 + b2*x + b3*x^2 + b4*x^3
    b2 = (sY1 * t11 + sY2 * t12 + sY3 * t13) / determinantS;
    b3 = (sY1 * t12 + sY2 * t22 + sY3 * t23) / determinantS;
    b4 = (sY1 * t13 + sY2 * t23 + sY3 * t33) / determinantS;
    b1 = YMean - b2 * XMean - b3 * XXMean - b4 * XXXMean;

    // Adjust for the bias
    if (independentVariable == enmIndependentVariable::X)
    {
        b1 = b1 - b4 * bias.x * bias.x * bias.x + b3 * bias.x * bias.x - b2 * bias.x + bias.y;
        b2 = b2 + 3.0f * b4 * bias.x * bias.x - 2.0f * b3 * bias.x;
        b3 = b3 - 3.0f * b4 * bias.x;
    }
    else
    {
        b1 = b1 - b4 * bias.y * bias.y * bias.y + b3 * bias.y * bias.y - b2 * bias.y + bias.x;
        b2 = b2 + 3.0f * b4 * bias.y * bias.y - 2.0f * b3 * bias.y;
        b3 = b3 - 3.0f * b4 * bias.y;
    }

    ValidRegressionModel = true;
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest1(vector<PointF>& points)
{
    //////////////////////////////////////
    // Unit test #1:  Vertical Parabola //
    //////////////////////////////////////

    // A cubic y = x^3 + 2 has the following points
    // [0 2]
    // [1 3]
    // [2 10]
    // [3 29]
    // 
    // We should be able to fit to these points and return the coefficients [2 0 0 1].

    points = vector<PointF>();
    points.push_back(PointF(0.0f, 2.0f));
    points.push_back(PointF(1.0f, 3.0f));
    points.push_back(PointF(2.0f, 10.0f));
    points.push_back(PointF(3.0f, 29.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest2(vector<PointF>& points)
{
    /////////////////////////////////////////////////
    // Unit test #1a:  Vertical Parabola with bias //
    /////////////////////////////////////////////////

    // A cubic y-400 = (x-500)^3 + 2 has the following points
    // [499 401]
    // [500 402]
    // [501 403]
    // [502 410]
    // [503 429]
    // 

    points = vector<PointF>();
    points.push_back(PointF(499.0f, 401.0f));
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 410.0f));
    points.push_back(PointF(503.0f, 429.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest3(vector<PointF>& points)
{
    ////////////////////
    // Unit test #2:  //
    ////////////////////

    // A cubic x = y^3 + y has the following points
    // [-2 -1]
    // [0 0]
    // [2 1]
    // [10 2]
    // 
    // We should be able to fit to these points and return the coefficients [0 1 0 1].

    points = vector<PointF>();
    points.push_back(PointF(-2.0f, -1.0f));
    points.push_back(PointF(0.0f, 0.0f));
    points.push_back(PointF(2.0f, 1.0f));
    points.push_back(PointF(10.0f, 2.0f));

    return CalculateCubicRegressionConsensus(points, enmIndependentVariable::Y);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest5(vector<PointF>& points)
{
    ///////////////////////////////////////////////////
    // Unit test #2a:  Horizontal Parabola with bias //
    ///////////////////////////////////////////////////

    // A simple horizontal parabola x - 400 = (y-500)^2 + (y-500) has the following points
    // [400 499]
    // [400 500]
    // [402 501]
    // [406 502]
    // 
    // We should be able to fit to these points and return the coefficients [249900 -999 1].

    points = vector<PointF>();
    points.push_back(PointF(400.0f, 499.0f));
    points.push_back(PointF(400.0f, 500.0f));
    points.push_back(PointF(402.0f, 501.0f));
    points.push_back(PointF(406.0f, 502.0f));

    return CalculateCubicRegressionConsensus(points, enmIndependentVariable::Y);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest4(vector<PointF>& points)
{
    ////////////////////
    // Unit test #2bias:  //
    ////////////////////

    // A cubic x = y^3 + y has the following points
    // [398 499]
    // [400 500]
    // [402 501]
    // [410 502]
    // 
    // We should be able to fit to these points and return the coefficients [-125000102 750001 -1500 1].

    points = vector<PointF>();
    points.push_back(PointF(398.0f, 499.0f));
    points.push_back(PointF(400.0f, 500.0f));
    points.push_back(PointF(402.0f, 501.0f));
    points.push_back(PointF(410.0f, 502.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest7(vector<PointF>& points)
{
    /////////////////////////////////////////////////
    // Unit test #1d:  Vertical Parabola with bias //
    /////////////////////////////////////////////////

    // A simple vertical parabola y - 400 = (x-500)^2 + 2 has the following points
    // [500 402]
    // [501 403]
    // [502 406]
    // [503 411]
    // 
    // We should be able to fit to these points and return the coefficients [250402 -1000 1 0].

    points = vector<PointF>();
    points.push_back(PointF(496.0f, 418.0f));
    points.push_back(PointF(497.0f - 1.2f, 411.0f));
    points.push_back(PointF(498.0f, 406.0f));
    points.push_back(PointF(499.0f, 403.0f));
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 406.0f));
    points.push_back(PointF(503.0f, 411.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest6(vector<PointF>& pointsPA)
{
    ////////////////////////////////////////////////////////
    // Unit test #3:  Left Bead From Pacific Amore Bottle //
    ////////////////////////////////////////////////////////

    pointsPA = vector<PointF>();
    pointsPA.push_back(PointF(433.00f, 593.f));
    pointsPA.push_back(PointF(432.00f, 594.f));
    pointsPA.push_back(PointF(431.50f, 595.f));
    pointsPA.push_back(PointF(430.70f, 596.f));
    pointsPA.push_back(PointF(430.56f, 597.f));
    pointsPA.push_back(PointF(430.55f, 598.f));
    pointsPA.push_back(PointF(430.70f, 599.f));
    pointsPA.push_back(PointF(431.50f, 600.f));
    pointsPA.push_back(PointF(432.40f, 601.f));
    pointsPA.push_back(PointF(434.01f, 602.f));
    pointsPA.push_back(PointF(436.01f, 603.f));

    return CalculateCubicRegressionConsensus(pointsPA, enmIndependentVariable::Y);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest8(vector<PointF>& points)
{
    points = vector<PointF>();
    points.push_back(PointF(496.0f, 418.0f));
    points.push_back(PointF(497.0f, 411.0f));
    points.push_back(PointF(498.0f, 406.0f));
    points.push_back(PointF(499.0f, 403.0f));
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 406.0f));
    points.push_back(PointF(503.0f + 3.0f, 411.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest9(vector<PointF>& pointsPAb)
{
    ////////////////////////////////////////////////////////
    // Unit test #3:  Left Bead From Pacific Amore Bottle //
    ////////////////////////////////////////////////////////

    // 
    // We should be able to fit to these points and return the coefficients [0 1 1].

    pointsPAb = vector<PointF>();
    pointsPAb.push_back(PointF(433.00f, 593.f));
    pointsPAb.push_back(PointF(432.00f, 594.f));
    pointsPAb.push_back(PointF(431.50f, 595.f));
    pointsPAb.push_back(PointF(430.70f, 596.f));
    pointsPAb.push_back(PointF(430.56f, 597.f));
    pointsPAb.push_back(PointF(430.55f, 598.f));
    pointsPAb.push_back(PointF(430.70f, 599.f));
    pointsPAb.push_back(PointF(431.50f, 600.f));
    pointsPAb.push_back(PointF(432.40f, 601.f));
    pointsPAb.push_back(PointF(434.01f, 602.f));
    pointsPAb.push_back(PointF(436.01f, 603.f));
    pointsPAb.push_back(PointF(437.01f, 604.f));
    pointsPAb.push_back(PointF(437.01f, 605.f));
    pointsPAb.push_back(PointF(437.01f, 606.f));
    pointsPAb.push_back(PointF(437.01f, 607.f));
    pointsPAb.push_back(PointF(437.01f, 608.f));

    return CalculateCubicRegressionConsensus(pointsPAb, enmIndependentVariable::Y);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest10(vector<PointF>& pointsPAc)
{
    ////////////////////////////////////////////////////////
    // Unit test #3:  Left Bead From Pacific Amore Bottle //
    ////////////////////////////////////////////////////////

    // 
    // We should be able to fit to these points and return the coefficients [0 1 1].

    pointsPAc = vector<PointF>();
    pointsPAc.push_back(PointF(433.00f, 593.f));
    pointsPAc.push_back(PointF(432.00f, 594.f));
    pointsPAc.push_back(PointF(431.50f, 595.f));
    pointsPAc.push_back(PointF(430.70f, 596.f));
    pointsPAc.push_back(PointF(430.56f, 597.f));
    pointsPAc.push_back(PointF(430.55f, 598.f));
    pointsPAc.push_back(PointF(430.70f, 599.f));
    pointsPAc.push_back(PointF(431.50f, 600.f));
    pointsPAc.push_back(PointF(432.40f, 601.f));
    pointsPAc.push_back(PointF(434.01f, 602.f));
    pointsPAc.push_back(PointF(436.01f, 603.f));
    pointsPAc.push_back(PointF(437.01f, 604.f));
    pointsPAc.push_back(PointF(437.01f, 605.f));
    pointsPAc.push_back(PointF(437.01f, 606.f));
    pointsPAc.push_back(PointF(437.01f, 607.f));
    pointsPAc.push_back(PointF(437.01f, 608.f));
    pointsPAc.push_back(PointF(432.01f, 593.f));
    pointsPAc.push_back(PointF(431.01f, 593.f));
    pointsPAc.push_back(PointF(430.01f, 593.f));

    return CalculateCubicRegressionConsensus(pointsPAc, enmIndependentVariable::Y);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest11(vector<PointF>& points)
{
    ////////////////////
    // Unit test #3a: //
    ////////////////////

    // A cubic y-400 = (x-500)^3 + 2 has the following points
    // [499 401]
    // [500 402]
    // [501 403]
    // [502 410]
    // [503 429]
    // 
    // We should be able to fit to these points and return the coefficients [2 0 0 1].

    points = vector<PointF>();
    points.push_back(PointF(499.0f, 401.0f));
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 410.0f));
    points.push_back(PointF(503.0f, 429.0f));

    return CalculateCubicRegressionConsensus(points);
}

CubicRegression::CubicConsensusModel& CubicRegression::UnitTest12(vector<PointF>& points)
{
    ////////////////////
    // Unit test #3b: //
    ////////////////////

    points = vector<PointF>();
    points.push_back(PointF(399.7f, 515.0f));
    points.push_back(PointF(400.4f, 514.0f));
    points.push_back(PointF(401.3f, 513.0f));
    points.push_back(PointF(402.4f, 512.0f));
    points.push_back(PointF(405.9f, 511.0f));
    points.push_back(PointF(412.0f, 510.0f));
    points.push_back(PointF(418.1f, 509.0f));
    points.push_back(PointF(419.1f, 508.0f));
    points.push_back(PointF(420.6f, 507.0f));
    points.push_back(PointF(420.5f, 506.0f));
    points.push_back(PointF(414.1f, 505.0f)); // lint
    points.push_back(PointF(413.9f, 504.0f)); // lint
    points.push_back(PointF(421.4f, 503.0f));

    return CalculateCubicRegressionConsensus(points);
}
