#include <iostream>
#include <string>
#include <vector>

#include "matplotlibcpp.h"

#include "LinearRegression.h"
#include "QuadraticRegression.h"
#include "CubicRegression.h"
#include "EllipticalRegression.h"

namespace plt = matplotlibcpp;

using namespace std;

void GetDataBounds(vector<PointF> points, double & minX, double & maxX, double & minY, double & maxY, vector<PointF> outliers = vector<PointF>())
{
    minX = 99999999.9;
    maxX = -99999999.9;
    minY = 99999999.9;
    maxY = -99999999.9;
    for (auto point: points)
    {
        for (auto outlier : outliers)
        {
            if (point.X == outlier.X && point.Y == outlier.Y)
            {
                continue;
            }
        }

        if (point.X < minX)
        {
            minX = point.X;
        }
        if (point.X > maxX)
        {
            maxX = point.X;
        }
        if (point.Y < minY)
        {
            minY = point.Y;
        }
        if (point.Y > maxY)
        {
            maxY = point.Y;
        }
    }
}

void GetChartBounds(vector<PointF> points, double & minXDisp, double & maxXDisp, double & minYDisp, double & maxYDisp, bool anscombes = false)
{
    double minX, maxX, minY, maxY;
    GetDataBounds(points, minX, maxX, minY, maxY);

    double displayBuffer = 2.0;
    minXDisp = minX - displayBuffer;
    maxXDisp = maxX + displayBuffer;
    minYDisp = minY - displayBuffer;
    maxYDisp = maxY + displayBuffer;

    if (anscombes)
    {
        // Special case
        minXDisp = 4;
        minYDisp = 2;
        maxXDisp = 20;
        maxYDisp = 14;
    }
}

vector<double> GetX(vector<PointF> points)
{
    vector<double> xvalues = vector<double>();
    for (auto p : points)
    {
        xvalues.push_back(p.X);
    }

    return xvalues;
}

vector<double> GetY(vector<PointF> points)
{
    vector<double> yvalues = vector<double>();
    for (auto p : points)
    {
        yvalues.push_back(p.Y);
    }

    return yvalues;
}

void DisplayRegressionLine(string title, vector<PointF> data, LinearRegression::LineModel& fit, vector<PointF> outliers, LinearRegression::LineModel& orig)
{
    // Title and data points
    double minX, maxX, minY, maxY;
    GetDataBounds(data, minX, maxX, minY, maxY/*, outliers*/);

    plt::title(title);
    plt::plot(GetX(data), GetY(data), { {"c", "blue"}, {"marker", "x"}, {"linestyle", ""}, {"label", "data points"} });
    plt::plot(GetX(outliers), GetY(outliers), { {"c", "red"}, {"marker", "+"}, {"linestyle", ""}, {"label", "outliers"} });

    // Axis limits
    double minXDisp, maxXDisp, minYDisp, maxYDisp;
    GetChartBounds(data, minXDisp, maxXDisp, minYDisp, maxYDisp, false /*title.Contains("nscombe")*/);
    plt::xlim(minXDisp, maxXDisp);
    plt::ylim(minYDisp, maxYDisp);

    // Consensus line
    if (fit.independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        plt::plot({ minXDisp, maxXDisp }, { fit.ModeledY((float)minXDisp), fit.ModeledY((float)maxXDisp) }, { {"c", "blue"}, {"marker", ""}, {"linestyle", "-"}, {"label", "consensus fit"} });
    }
    else
    {
        plt::plot({ fit.ModeledX((float)minYDisp), fit.ModeledX((float)maxYDisp) }, { minYDisp, maxYDisp }, { {"c", "blue"}, {"marker", ""}, {"linestyle", "-"}, {"label", "consensus fit"} });
    }

    // Original line
    if (orig.independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        plt::plot({ minXDisp, maxXDisp }, { orig.ModeledY((float)minXDisp), orig.ModeledY((float)maxXDisp) }, { {"c", "red"}, {"marker", ""}, {"linestyle", "-"}, {"label", "least squares regression"} });
    }
    else
    {
        plt::plot({ orig.ModeledX((float)minYDisp), orig.ModeledX((float)maxYDisp) }, { minYDisp, maxYDisp }, { {"c", "red"}, {"marker", ""}, {"linestyle", "-"}, {"label", "least squares regression"} });
    }

    plt::legend();
    plt::show();
}

void DisplayRegressionPoly(string title, vector<PointF> data, PolynomialModel& fit, vector<PointF> outliers, PolynomialModel& orig)
{
    // Title and data points
    double minX, maxX, minY, maxY;
    GetDataBounds(data, minX, maxX, minY, maxY/*, outliers*/);

    plt::title(title);
    plt::plot(GetX(data), GetY(data), { {"c", "blue"}, {"marker", "x"}, {"linestyle", ""}, {"label", "data points"} });
    plt::plot(GetX(outliers), GetY(outliers), { {"c", "red"}, {"marker", "+"}, {"linestyle", ""}, {"label", "outliers"} });

    // Axis limits
    double minXDisp, maxXDisp, minYDisp, maxYDisp;
    GetChartBounds(data, minXDisp, maxXDisp, minYDisp, maxYDisp, false /*title.Contains("nscombe")*/);
    plt::xlim(minXDisp, maxXDisp);
    plt::ylim(minYDisp, maxYDisp);

    // Consensus line
    double numPoints = 30.0;
    vector<double> xv = vector<double>();
    vector<double> yv = vector<double>();

    // Original line
    if (orig.independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        for (double x = minX; x <= maxX + 0.0001; x += (maxX - minX) / (numPoints - 1.0))
        {
            xv.push_back(x);
            yv.push_back(orig.ModeledY((float)x));
        }
    }
    else
    {
        for (double y = minY; y <= maxY + 0.0001; y += (maxY - minY) / (numPoints - 1.0))
        {
            yv.push_back(y);
            xv.push_back(orig.ModeledX((float)y));
        }
    }
    plt::plot(xv, yv, { {"c", "red"}, {"marker", ""}, {"linestyle", "-"}, {"label", "least squares regression"} });

    xv.clear();
    yv.clear();
    if (fit.independentVariable == PolynomialModel::enmIndependentVariable::X)
    {
        for (double x = minX; x <= maxX + 0.0001; x += (maxX - minX) / (numPoints - 1.0))
        {
            xv.push_back(x);
            yv.push_back(fit.ModeledY((float)x));
        }
    }
    else
    {
        for (double y = minY; y <= maxY + 0.0001; y += (maxY - minY) / (numPoints - 1.0))
        {
            yv.push_back(y);
            xv.push_back(fit.ModeledX((float)y));
        }
    }
    plt::plot(xv, yv, { {"c", "blue"}, {"marker", ""}, {"linestyle", "-"}, {"label", "consensus fit"} });

    plt::legend();
    plt::show();
}

void DisplayRegressionEllipse(string title, vector<PointF> data, EllipticalRegression::EllipseModel& model, vector<PointF> outliers, EllipticalRegression::EllipseModel& orig)
{
    // Title and data points
    double minX, maxX, minY, maxY;
    GetDataBounds(data, minX, maxX, minY, maxY, outliers);

    plt::title(title);
    plt::plot(GetX(data), GetY(data), { {"c", "blue"}, {"marker", "x"}, {"linestyle", ""}, {"label", "data points"} });
    plt::plot(GetX(outliers), GetY(outliers), { {"c", "red"}, {"marker", "+"}, {"linestyle", ""}, {"label", "outliers"} });

    // Axis limits
    double minXDisp, maxXDisp, minYDisp, maxYDisp;
    GetChartBounds(data, minXDisp, maxXDisp, minYDisp, maxYDisp, false /*title.Contains("nscombe")*/);
    plt::xlim(minXDisp, maxXDisp);
    plt::ylim(minYDisp, maxYDisp);

    double numPoints = 1000.0;
    double ellipseMinX = orig.x0 - orig.long_axis / 2.0;
    double ellipseMaxX = orig.x0 + orig.long_axis / 2.0;

    vector<double> xv = vector<double>();
    vector<double> yv = vector<double>();

    // Original
    for (double x = ellipseMinX; x <= ellipseMaxX + 0.0001; x += (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
    {
        auto modeledY = EllipticalRegression::ModeledY(orig, (float)x, EllipticalRegression::EllipseHalves::TopHalf);
        if (modeledY > minYDisp && modeledY < maxYDisp)
        {
            xv.push_back(x);
            yv.push_back(modeledY);
        }
    }
    for (double x = ellipseMaxX; x >= ellipseMinX - 0.0001; x -= (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
    {
        auto modeledY = EllipticalRegression::ModeledY(orig, (float)x, EllipticalRegression::EllipseHalves::BottomHalf);
        if (modeledY > minYDisp && modeledY < maxYDisp)
        {
            xv.push_back(x);
            yv.push_back(modeledY);
        }
    }
    plt::plot(xv, yv, { {"c", "red"}, {"marker", ""}, {"linestyle", "-"}, {"label", "least squares regression"} });

    xv.clear();
    yv.clear();

    // Consensus
    for (double x = ellipseMinX; x <= ellipseMaxX + 0.0001; x += (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
    {
        auto modeledY = EllipticalRegression::ModeledY(model, (float)x, EllipticalRegression::EllipseHalves::TopHalf);
        if (modeledY > minYDisp && modeledY < maxYDisp)
        {
            xv.push_back(x);
            yv.push_back(modeledY);
        }
    }
    for (double x = ellipseMaxX; x >= ellipseMinX - 0.0001; x -= (ellipseMaxX - ellipseMinX) / (numPoints - 1.0))
    {
        auto modeledY = EllipticalRegression::ModeledY(model, (float)x, EllipticalRegression::EllipseHalves::BottomHalf);
        if (modeledY > minYDisp && modeledY < maxYDisp)
        {
            xv.push_back(x);
            yv.push_back(modeledY);
        }
    }
    plt::plot(xv, yv, { {"c", "blue"}, {"marker", ""}, {"linestyle", "-"}, {"label", "consensus fit"} });

    plt::legend();
    plt::show();
}

int main(int argc, char** argv)
{
    vector<PointF> points, outliers;
    LinearRegression::LinearConsensusModel consensus = LinearRegression::LinearConsensusModel(PolynomialModel::enmIndependentVariable::X);
    QuadraticRegression::QuadraticConsensusModel qconsensus = QuadraticRegression::QuadraticConsensusModel(PolynomialModel::enmIndependentVariable::X);
    CubicRegression::CubicConsensusModel cconsensus = CubicRegression::CubicConsensusModel(PolynomialModel::enmIndependentVariable::X);
    EllipticalRegression::EllipseConsensusModel econsensus = EllipticalRegression::EllipseConsensusModel();
    
    consensus = LinearRegression::UnitTest4(points);
    DisplayRegressionLine("Consensus Regression splits data points into inliers and outliers\nClose figure to see the next", points, static_cast<LinearRegression::LineModel&>(*consensus.model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.original));
    consensus = LinearRegression::UnitTest5(points);
    DisplayRegressionLine("Linear Regression - Corner", points, static_cast<LinearRegression::LineModel&>(*consensus.model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.original));

    consensus = LinearRegression::UnitTestA1(points);
    DisplayRegressionLine("Linear Regression A1 - Anscombe's Quartet", points, static_cast<LinearRegression::LineModel&>(*consensus.model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.original));
    consensus = LinearRegression::UnitTestA3(points);
    DisplayRegressionLine("Linear Regression A3 - Anscombe's Quartet", points, static_cast<LinearRegression::LineModel&>(*consensus.model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.original));
    consensus = LinearRegression::UnitTestA4(points);
    DisplayRegressionLine("Linear Regression A4 - Anscombe's Quartet", points, static_cast<LinearRegression::LineModel&>(*consensus.model), consensus.Outliers, static_cast<LinearRegression::LineModel&>(*consensus.original));
    consensus = LinearRegression::UnitTestA2(points);
    DisplayRegressionLine("Linear Regression A2 - Anscombe's Quartet", points, static_cast<LinearRegression::LineModel&>(*consensus.original), vector<PointF>(), static_cast<LinearRegression::LineModel&>(*consensus.original));
    
    qconsensus = QuadraticRegression::UnitTestA2(points);
    DisplayRegressionPoly("Quadratic Regression A2 - Anscombe's Quartet", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest1(points);
    DisplayRegressionPoly("Quadratic Test 1 - Vertical Parabola, no outliers", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest3(points);
    DisplayRegressionPoly("Quadratic Test 2 - Horizontal Parabola, no outliers", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest5(points);
    DisplayRegressionPoly("Quadratic Test 3 - One outlier", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest6(points);
    DisplayRegressionPoly("Quadratic Test 4 - One outlier", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest7(points);
    DisplayRegressionPoly("Quadratic Test 5 - Real data", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest8(points);
    DisplayRegressionPoly("Quadratic Test 6 - Real data, plus syn outliers", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    qconsensus = QuadraticRegression::UnitTest9(points);
    DisplayRegressionPoly("Quadratic Test 7 - Real data, plus syn outliers", points, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.model), qconsensus.outliers, static_cast<QuadraticRegression::QuadraticModel&>(*qconsensus.original));
    
    cconsensus = CubicRegression::UnitTest2(points);
    DisplayRegressionPoly("Cubic Test 1 - No outliers", points, static_cast<CubicRegression::CubicModel&>(*cconsensus.model), cconsensus.outliers, static_cast<CubicRegression::CubicModel&>(*cconsensus.original));
    cconsensus = CubicRegression::UnitTest3(points);
    DisplayRegressionPoly("Cubic Test 2 - No outliers, y-independent", points, static_cast<CubicRegression::CubicModel&>(*cconsensus.model), cconsensus.outliers, static_cast<CubicRegression::CubicModel&>(*cconsensus.original));
    cconsensus = CubicRegression::UnitTest12(points);
    DisplayRegressionPoly("Cubic Test 3 - real data", points, static_cast<CubicRegression::CubicModel&>(*cconsensus.model), cconsensus.outliers, static_cast<CubicRegression::CubicModel&>(*cconsensus.original));
    cconsensus = CubicRegression::UnitTest8(points);
    DisplayRegressionPoly("Cubic Test 4 - cubic to quadratic data", points, static_cast<CubicRegression::CubicModel&>(*cconsensus.model), cconsensus.outliers, static_cast<CubicRegression::CubicModel&>(*cconsensus.original));

    econsensus = EllipticalRegression::UnitTest1(points);
    DisplayRegressionEllipse("Ellipse Test 1 - No outliers", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));
    econsensus = EllipticalRegression::UnitTest2(points);
    DisplayRegressionEllipse("Ellipse Test 2", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));
    econsensus = EllipticalRegression::UnitTest3(points);
    DisplayRegressionEllipse("Ellipse Test 3", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));
    econsensus = EllipticalRegression::UnitTest4(points);
    DisplayRegressionEllipse("Ellipse Test 4", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));
    econsensus = EllipticalRegression::UnitTest5(points);
    DisplayRegressionEllipse("Ellipse Test 5", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));
    econsensus = EllipticalRegression::UnitTest6(points);
    DisplayRegressionEllipse("Ellipse Test 6", points, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.model), econsensus.outliers, static_cast<EllipticalRegression::EllipseModel&>(*econsensus.original));

    return 0;
}

