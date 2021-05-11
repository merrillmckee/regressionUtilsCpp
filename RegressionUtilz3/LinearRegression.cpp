#include "LinearRegression.h"

const float LinearRegression::DEFAULT_SENSITIVITY = 0.2f;

float LinearRegression::LineModel::ModeledY(float x)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::X)
    {
        return (float)(b1 + b2 * (double)x);
    }
    else
    {
        return -99999999.9f;
    }
}

float LinearRegression::LineModel::ModeledX(float y)
{
    if (ValidRegressionModel && independentVariable == enmIndependentVariable::Y)
    {
        return (float)(b1 + b2 * (double)y);
    }
    else
    {
        return -99999999.9f;
    }
}

LinearRegression::LinearConsensusModel& LinearRegression::CalculateLinearRegressionConsensus(vector<PointF> points, PolynomialModel::enmIndependentVariable independentVariable, float sensitivity)
{
    auto consensus = new LinearConsensusModel(independentVariable);
    consensus->Calculate(points, sensitivity);

    return *consensus;
}

float LinearRegression::LinearConsensusModel::CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide)
{
    LineModel& line = static_cast<LineModel&>(model);

    if (point.IsEmpty)
    {
        pointOnPositiveSide = false;
        return 999999999.9f;
    }

    if (line.independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        auto numerator = -line.b2 * point.X + point.Y - line.b1;

        pointOnPositiveSide = true;
        if (numerator < 0.0f)
        {
            pointOnPositiveSide = false;
        }

        return (float)(abs(numerator) / sqrt(line.b2 * line.b2 + 1.0));
    }
    else
    {
        auto numerator = -line.b2 * point.Y + point.X - line.b1;

        pointOnPositiveSide = true;
        if (numerator < 0.0f)
        {
            pointOnPositiveSide = false;
        }

        return (float)(abs(numerator) / sqrt(line.b2 * line.b2 + 1.0));
    }
}

RegressionModel::Summations* LinearRegression::LineModel::CalculateSummations(vector<PointF> points)
{
    LinearSummations* sum = new LinearSummations();
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

    // Shorthand that better matches the math formulas
    auto N = sum->N = (int)points.size();

    // Calculate the summations
    for (auto i = 0; i < N; ++i)
    {
        // Shorthand
        auto x = points[i].X;
        auto y = points[i].Y;

        // Meh
        if (independentVariable == PolynomialModel::enmIndependentVariable::Y)
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
    }

    return sum;
}

void LinearRegression::LineModel::CalculateFeatures()
{
    slope = b2;
    intercept = b1;
}

void LinearRegression::LineModel::CalculateModel(Summations& sums)
{
    if (sums.N <= 0)
    {
        ValidRegressionModel = false;
        return;
    }

    LinearSummations& sum = static_cast<LinearSummations&>(sums);

    // Calculate the means
    auto XMean = sum.x / (double)sum.N;
    auto YMean = sum.y / (double)sum.N;

    // Calculate the S intermediate values
    auto s11 = sum.x2 - (1.0 / (double)sum.N) * sum.x * sum.x;
    auto sY1 = sum.xy - (1.0 / (double)sum.N) * sum.x * sum.y;

    // Don't divide by zero
    // Note:  Maintaining the matrix notation even though S or s11 is a 1x1 "matrix".  For higher degrees, 
    //        the notation will remain consistent.
    auto determinantS = s11;
    if (abs(determinantS) <= RegressionModel::EPSILON)
    {
        ValidRegressionModel = false;
        return;
    }

    // Calculate the coefficients of y = b1 + b2*x
    b2 = sY1 / determinantS;
    b1 = YMean - b2 * XMean;

    // Adjust for the bias
    if (independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        b1 = b1 + bias.y - b2 * bias.x;
    }
    else
    {
        b1 = b1 + bias.x - b2 * bias.y;
    }

    ValidRegressionModel = true;
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTestA1(vector<PointF> & anscombe1)
{
    // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

    anscombe1 = vector<PointF>();
    anscombe1.push_back(PointF(10.0f, 8.04f));
    anscombe1.push_back(PointF(8.0f, 6.95f));
    anscombe1.push_back(PointF(13.0f, 7.58f));
    anscombe1.push_back(PointF(9.0f, 8.81f));
    anscombe1.push_back(PointF(11.0f, 8.33f));
    anscombe1.push_back(PointF(14.0f, 9.96f));
    anscombe1.push_back(PointF(6.0f, 7.24f));
    anscombe1.push_back(PointF(4.0f, 4.26f));
    anscombe1.push_back(PointF(12.0f, 10.84f));
    anscombe1.push_back(PointF(7.0f, 4.82f));
    anscombe1.push_back(PointF(5.0f, 5.68f));

    return CalculateLinearRegressionConsensus(anscombe1);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTestA2(vector<PointF>& anscombe2)
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

    return CalculateLinearRegressionConsensus(anscombe2);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTestA3(vector<PointF>& anscombe3)
{
    // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

    anscombe3 = vector<PointF>();
    anscombe3.push_back(PointF(10.0f, 7.46f));
    anscombe3.push_back(PointF(8.0f, 6.77f));
    anscombe3.push_back(PointF(13.0f, 12.74f));
    anscombe3.push_back(PointF(9.0f, 7.11f));
    anscombe3.push_back(PointF(11.0f, 7.81f));
    anscombe3.push_back(PointF(14.0f, 8.84f));
    anscombe3.push_back(PointF(6.0f, 6.08f));
    anscombe3.push_back(PointF(4.0f, 5.39f));
    anscombe3.push_back(PointF(12.0f, 8.15f));
    anscombe3.push_back(PointF(7.0f, 6.42f));
    anscombe3.push_back(PointF(5.0f, 5.73f));

    return CalculateLinearRegressionConsensus(anscombe3);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTestA4(vector<PointF>& anscombe)
{
    // Anscombe's quartet - https://en.wikipedia.org/wiki/Anscombe%27s_quartet

    anscombe = vector<PointF>();
    anscombe.push_back(PointF(8.01f, 5.25f));
    anscombe.push_back(PointF(8.02f, 5.56f));
    anscombe.push_back(PointF(8.03f, 5.76f));
    anscombe.push_back(PointF(8.04f, 6.58f));
    anscombe.push_back(PointF(8.05f, 6.89f));
    anscombe.push_back(PointF(8.06f, 7.71f));
    anscombe.push_back(PointF(8.07f, 7.91f));
    anscombe.push_back(PointF(8.08f, 8.47f));
    anscombe.push_back(PointF(8.09f, 8.84f));
    anscombe.push_back(PointF(8.05f, 7.04f));
    anscombe.push_back(PointF(19.0f, 12.5f));

    return CalculateLinearRegressionConsensus(anscombe);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTest1(vector<PointF>& points)
{
    ////////////////////////////////////////
    // Unit test #1:  Line with slope = 2 //
    ////////////////////////////////////////

    // A line y = 2x + 1 has the following points
    // [0 1]
    // [1 3]
    // [2 5]
    // 
    // We should be able to fit to these points and return the coefficients [1 2].

    points = vector<PointF>();
    points.push_back(PointF(-3.0f, -5.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-2.0f, -3.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.5f, -2.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.0f, -1.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-0.5f, 0.0f));  // True line point:  y = 2x + 1
    points.push_back(PointF(0.0f, 1.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(0.5f, 2.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(1.0f, 3.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(2.0f, 5.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(3.0f, 9.5f));   // <--- Adding in 2.5 noise
    points.push_back(PointF(4.0f, 7.0f));   // <--- Adding in -2.0 noise
    points.push_back(PointF(5.0f, 14.5f));  // <--- Adding in 3.5 noise
    points.push_back(PointF(5.0f, 11.0f));  // True line point:  y = 2x + 1
    points.push_back(PointF(7.0f, 11.0f));  // <--- Adding in 4.0 noise

    return CalculateLinearRegressionConsensus(points);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTest2(vector<PointF>& points)
{
    //////////////////////////////////
    // Unit test #2:  Vertical line //
    //////////////////////////////////

    // A simple vertical line x = 3 has the following points
    // [3 0]
    // [3 1]
    // [3 2]
    // 
    // We should be able to fit to these points and return the coefficients [3 0].

    points = vector<PointF>();
    points.push_back(PointF(3.0f, 0.0f));
    points.push_back(PointF(3.0f, 1.0f));
    points.push_back(PointF(3.0f, 2.0f));

    return CalculateLinearRegressionConsensus(points);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTest3(vector<PointF>& points)
{
    //////////////////////////////////////////////////
    // Unit test #3 with bias:  Line with slope = 2 //
    //////////////////////////////////////////////////

    // A line y-11000 = 2(x-1500) + 1 ... OR ... y = 2x + 8001 ... has the following points
    // [1500 11001]
    // [1501 11003]
    // [1502 11005]
    // 
    // We should be able to fit to these points and return the coefficients [1 2].

    points = vector<PointF>();
    points.push_back(PointF(1500.0f, 11001.0f));
    points.push_back(PointF(1501.0f, 11003.0f));
    points.push_back(PointF(1502.0f, 11005.0f));
    points.push_back(PointF(1503.0f, 11007.0f));
    points.push_back(PointF(1504.0f, 11009.0f));
    points.push_back(PointF(1505.0f, 11011.0f));
    points.push_back(PointF(1506.0f, 11013.0f));

    return CalculateLinearRegressionConsensus(points);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTest4(vector<PointF>& points)
{
    ////////////////////////////////////////
    // Unit test #4:  Line with slope = 2 //
    ////////////////////////////////////////

    // A line y = 2x + 1 has the following points
    // [0 1]
    // [1 3]
    // [2 5]
    // 
    // We should be able to fit to these points and return the coefficients [1 2].

    // Add some random noise to test the stopping condition
    points = vector<PointF>();
    points.push_back(PointF(-3.0f + rand() % 100 / 400.0f - 0.125f, -5.0f + rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-2.0f + rand() % 100 / 400.0f - 0.125f, -3.0f + rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.5f + rand() % 100 / 400.0f - 0.125f, -2.0f + rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.0f + rand() % 100 / 400.0f - 0.125f, -1.0f + rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-0.5f + rand() % 100 / 400.0f - 0.125f, 0.0f +  rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(0.0f +  rand() % 100 / 400.0f - 0.125f, 1.0f +  rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(0.5f +  rand() % 100 / 400.0f - 0.125f, 2.0f +  rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(1.0f +  rand() % 100 / 400.0f - 0.125f, 3.0f +  rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(2.0f +  rand() % 100 / 400.0f - 0.125f, 5.0f +  rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(3.0f +  rand() % 100 / 400.0f - 0.125f, 9.5f +  rand() % 100 / 400.0f - 0.125f)); // <--- Adding in 2.5 noise
    points.push_back(PointF(4.0f +  rand() % 100 / 400.0f - 0.125f, 7.0f +  rand() % 100 / 400.0f - 0.125f)); // <--- Adding in -2.0 noise
    points.push_back(PointF(5.0f +  rand() % 100 / 400.0f - 0.125f, 7.5f +  rand() % 100 / 400.0f - 0.125f)); // <--- Adding in -3.5 noise
    points.push_back(PointF(5.0f +  rand() % 100 / 400.0f - 0.125f, 11.0f + rand() % 100 / 400.0f - 0.125f)); // True line point:  y = 2x + 1
    points.push_back(PointF(7.0f +  rand() % 100 / 400.0f - 0.125f, 13.0f + rand() % 100 / 400.0f - 0.125f)); // <--- Adding in -2.0 noise

    return CalculateLinearRegressionConsensus(points);
}

LinearRegression::LinearConsensusModel& LinearRegression::UnitTest5(vector<PointF>& points)
{
    /////////////////////////////////////////////////////////////////////////////
    // Unit test #4:  Line with slope = 2 meets another line (corner scenario) //
    /////////////////////////////////////////////////////////////////////////////

    // A line y = 2x + 1 has the following points
    // [0 1]
    // [1 3]
    // [2 5]
    // 
    // A line y = -x + 16 intersects the first line at (5,11)
    //

    points = vector<PointF>();
    points.push_back(PointF(-3.0f, -5.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-2.0f, -3.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.5f, -2.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-1.0f, -1.0f)); // True line point:  y = 2x + 1
    points.push_back(PointF(-0.5f, 0.0f));  // True line point:  y = 2x + 1
    points.push_back(PointF(0.0f, 1.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(0.5f, 2.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(1.0f, 3.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(2.0f, 5.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(3.0f, 7.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(4.0f, 9.0f));   // True line point:  y = 2x + 1
    points.push_back(PointF(5.0f, 11.0f));  // True line point:  y = 2x + 1
    points.push_back(PointF(6.0f, 10.0f));  // True line point:  y = -x + 16
    points.push_back(PointF(7.0f, 9.0f));   // True line point:  y = -x + 16
    points.push_back(PointF(8.0f, 8.0f));   // True line point:  y = -x + 16
    points.push_back(PointF(9.0f, 7.0f));   // True line point:  y = -x + 16

    return CalculateLinearRegressionConsensus(points);
}

//int main(int argc, char** argv)
//{
//    vector<PointF> points, outliers;
//    LinearRegression::LinearConsensusModel consensus = LinearRegression::LinearConsensusModel(PolynomialModel::enmIndependentVariable::X);
//
//    consensus = static_cast<LinearRegression::LinearConsensusModel&>(LinearRegression::UnitTest4(points));
//    //DisplayRegressionLine("Consensus Regression splits data points into inliers and outliers\nClose figure to see the next", points, static_cast<LinearRegression::LineModel&>(*consensus.Model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.Original));
//}
