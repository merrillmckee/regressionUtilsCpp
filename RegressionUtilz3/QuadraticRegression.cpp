#include "QuadraticRegression.h"

const float QuadraticRegression::DEFAULT_SENSITIVITY = 0.35f;

float QuadraticRegression::QuadraticModel::ModeledY(float x)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::X)
    {
        return (float)(b1 + b2 * x + b3 * x * x);
    }
    else
    {
        return -99999999.9f;
    }
}

float QuadraticRegression::QuadraticModel::ModeledX(float y)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::Y)
    {
        return (float)(b1 + b2 * y + b3 * y * y);
    }
    else
    {
        return -99999999.9f;
    }
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::CalculateQuadraticRegressionConsensus(vector<PointF> points, enmIndependentVariable independentVariable, float sensitivity)
{
    auto consensus = new QuadraticConsensusModel(independentVariable);
    consensus->Calculate(points, sensitivity);

    return *consensus;
}

float QuadraticRegression::QuadraticConsensusModel::CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide)
{
    if (point.IsEmpty)
    {
        pointOnPositiveSide = false;
        return 99999999.9f;
    }

    auto error = 0.0f;
    auto qmodel = static_cast<QuadraticModel&>(model);
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

RegressionModel::Summations* QuadraticRegression::QuadraticModel::CalculateSummations(vector<PointF> points)
{
    QuadraticSummations* sum = new QuadraticSummations();
    if (points.size() < MinimumPoints)
    {
        sum->N = 0;
        return sum;
    }

    // Initialize all the summations to zero
    sum->x = 0.0;
    sum->y = 0.0;
    sum->x2 = 0.0;
    sum->xy = 0.0;
    sum->x3 = 0.0;
    sum->x2y = 0.0;  // (i.e.  SUM(x^2*y))
    sum->x4 = 0.0;

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

        // Sums
        sum->x += x;
        sum->y += y;
        sum->x2 += xx;
        sum->xy += xy;
        sum->x3 += x * xx;
        sum->x2y += xx * y;
        sum->x4 += xx * xx;
    }

    return sum;
}

void QuadraticRegression::QuadraticModel::CalculateFeatures()
{
    // Don't divide by zero when calculating the vertex of the parabola
    // If there is a division-by-zero when calculating the vertex of the 
    // parabola, it means the model is near-linear.  The coefficient of the 
    // squared term is near-zero.
    //if (Math.Abs(b3) <= EPSILON)
    //{
    //    x0 = float.MinValue;
    //    y0 = float.MinValue;
    //}
    //else
    //{
    //    // Calculate the vertex of the parabola
    //    x0 = (float)(b2 / (-2.0 * b3));
    //    y0 = (float)(b1 - (b2 * b2) / (4.0 * b3));
    //}
}

void QuadraticRegression::QuadraticModel::CalculateModel(Summations& sums)
{
    if (sums.N <= 0)
    {
        ValidRegressionModel = false;
        return;
    }

    QuadraticSummations& sum = static_cast<QuadraticSummations&>(sums);

    // Calculate the means
    auto XMean = sum.x / (double)sum.N;
    auto YMean = sum.y / (double)sum.N;
    auto XXMean = sum.x2 / (float)sum.N;

    // Calculate the S intermediate values
    auto s11 = sum.x2 - (1.0 / (double)sum.N) * sum.x * sum.x;
    auto s12 = sum.x3 - (1.0 / (double)sum.N) * sum.x * sum.x2;
    auto s22 = sum.x4 - (1.0 / (double)sum.N) * sum.x2 * sum.x2;
    auto sY1 = sum.xy - (1.0 / (double)sum.N) * sum.x * sum.y;
    auto sY2 = sum.x2y - (1.0 / (double)sum.N) * sum.x2 * sum.y;

    // Don't divide by zero
    auto determinantS = s22 * s11 - s12 * s12;
    if (abs(determinantS) <= EPSILON)
    {
        ValidRegressionModel = false;
        return;
    }

    // Calculate the coefficients of y = b1 + b2*x + b3*x^2
    b2 = (sY1 * s22 - sY2 * s12) / determinantS;
    b3 = (sY2 * s11 - sY1 * s12) / determinantS;
    b1 = YMean - b2 * XMean - b3 * XXMean;

    // Adjust for the bias
    if (independentVariable == enmIndependentVariable::X)
    {
        b1 = b1 + bias.y + b3 * bias.x * bias.x - b2 * bias.x;
        b2 = b2 - 2.0f * b3 * bias.x;
    }
    else
    {
        b1 = b1 + bias.x + b3 * bias.y * bias.y - b2 * bias.y;
        b2 = b2 - 2.0f * b3 * bias.y;
    }

    ValidRegressionModel = true;
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTestA2(vector<PointF>& anscombe2)
{
    // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

    anscombe2 = vector<PointF>();
    anscombe2.push_back(PointF(10.0f, 9.14f));
    anscombe2.push_back(PointF(8.0f, 8.14f));
    anscombe2.push_back(PointF(13.0f, 8.74f));
    anscombe2.push_back(PointF(9.0f, 8.77f));
    anscombe2.push_back(PointF(11.0f, 9.26f));
    anscombe2.push_back(PointF(14.0f, 8.10f));
    anscombe2.push_back(PointF(6.0f, 6.13f));
    anscombe2.push_back(PointF(4.0f, 3.10f));
    anscombe2.push_back(PointF(12.0f, 9.13f));
    anscombe2.push_back(PointF(7.0f, 7.26f));
    anscombe2.push_back(PointF(5.0f, 4.74f));

    return CalculateQuadraticRegressionConsensus(anscombe2);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest1(vector<PointF>& points)
{
    //////////////////////////////////////
    // Unit test #1:  Vertical Parabola //
    //////////////////////////////////////

    // A simple vertical parabola y = x^2 + 2 has the following points
    // [0 2]
    // [1 3]
    // [2 6]
    // [3 11]
    // 
    // We should be able to fit to these points and return the coefficients [2 0 1].

    points = vector<PointF>();
    points.push_back(PointF(0.0f, 2.0f));
    points.push_back(PointF(1.0f, 3.0f));
    points.push_back(PointF(2.0f, 6.0f));
    points.push_back(PointF(3.0f, 11.0f));

    return CalculateQuadraticRegressionConsensus(points);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest2(vector<PointF>& points)
{
    /////////////////////////////////////////////////
    // Unit test #1a:  Vertical Parabola with bias //
    /////////////////////////////////////////////////

    // A simple vertical parabola y - 400 = (x-500)^2 + 2 has the following points
    // [500 402]
    // [501 403]
    // [502 406]
    // [503 411]
    // 
    // We should be able to fit to these points and return the coefficients [250402 -1000 1].

    points = vector<PointF>();
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 406.0f));
    points.push_back(PointF(503.0f, 411.0f));

    return CalculateQuadraticRegressionConsensus(points);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest3(vector<PointF>& points)
{
    ////////////////////////////////////////
    // Unit test #2:  Horizontal Parabola //
    ////////////////////////////////////////

    // A simple horizontal parabola x = y^2 + y has the following points
    // [0 -1]
    // [0 0]
    // [2 1]
    // 
    // We should be able to fit to these points and return the coefficients [0 1 1].

    points = vector<PointF>();
    points.push_back(PointF(0.0f, -1.0f));
    points.push_back(PointF(0.0f, 0.0f));
    points.push_back(PointF(2.0f, 1.0f));

    return CalculateQuadraticRegressionConsensus(points, enmIndependentVariable::Y);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest4(vector<PointF>& points)
{
    ///////////////////////////////////////////////////
    // Unit test #2a:  Horizontal Parabola with bias //
    ///////////////////////////////////////////////////

    // A simple horizontal parabola x - 400 = (y-500)^2 + (y-500) has the following points
    // [400 499]
    // [400 500]
    // [402 501]
    // 
    // We should be able to fit to these points and return the coefficients [249900 -999 1].

    points = vector<PointF>();
    points.push_back(PointF(400.0f, 499.0f));
    points.push_back(PointF(400.0f, 500.0f));
    points.push_back(PointF(402.0f, 501.0f));

    return CalculateQuadraticRegressionConsensus(points, enmIndependentVariable::Y);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest5(vector<PointF>& points)
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
    // We should be able to fit to these points and return the coefficients [250402 -1000 1].

    points = vector<PointF>();
    points.push_back(PointF(496.0f, 418.0f));
    points.push_back(PointF(497.0f - 1.2f, 411.0f));
    points.push_back(PointF(498.0f, 406.0f));
    points.push_back(PointF(499.0f, 403.0f));
    points.push_back(PointF(500.0f, 402.0f));
    points.push_back(PointF(501.0f, 403.0f));
    points.push_back(PointF(502.0f, 406.0f));
    points.push_back(PointF(503.0f, 411.0f));

    return CalculateQuadraticRegressionConsensus(points);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest6(vector<PointF>& points)
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

    return CalculateQuadraticRegressionConsensus(points);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest7(vector<PointF>& pointsPA)
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

    return CalculateQuadraticRegressionConsensus(pointsPA, enmIndependentVariable::Y);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest8(vector<PointF>& pointsPAb)
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

    return CalculateQuadraticRegressionConsensus(pointsPAb, enmIndependentVariable::Y);
}

QuadraticRegression::QuadraticConsensusModel& QuadraticRegression::UnitTest9(vector<PointF>& pointsPAc)
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

    return CalculateQuadraticRegressionConsensus(pointsPAc, enmIndependentVariable::Y);
}
