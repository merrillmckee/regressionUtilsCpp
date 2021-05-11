#pragma once
#include <iostream>
#include <string>

#include "PolynomialRegression.h"
#include "RegressionConsensusModel.h"

using namespace std;

/// <summary>
/// QuadraticRegression
/// Author: Merrill McKee
/// Description:  The purpose of this class is to find the parabola that best fits through a set of 2D 
///   X, Y points.  It uses Quadratic Regression or the Least Squares Fit method to do this.  An array
///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
///   find the variables b1, b2, and b3 in the equation y = b1 + b2*x + b3*x^2.  The vertex of this 
///   parabola is (x0, y0).  For more information on the formulas used, see the website 
///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression.
///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
///   and ModeledY to get the X value for any given Y position along the parabola or get the Y value for any given
///   X position along the parabola.
///   
///   Note:  Vertically oriented parabolas assume an independent x-value.  Horizontally oriented parabolas 
///          assume an independent y-value.  Internally, a horizontally oriented parabola swaps the 
///          x and y values but the user's interface should not be affected.
///   
///   Notes: Additional matrix algebra details not in the website link:
///          
///          [b2] = [s11 s12]-1 * [sY1]
///          [b3]   [s12 s22]     [sY2]
///          
///          [b2] = (1 / det(S)) [ s22 -s12] * [sY1]
///          [b3]                [-s12  s11]   [sY2]
///          
///          This derivation will allow us to use this same algorithm for cubic regression.  The inverse of a 3x3 
///          matrix is a bit more tedious
/// </summary>

class QuadraticRegression : public PolynomialModel
{
protected:
    const static float DEFAULT_SENSITIVITY;

public:
    class QuadraticModel : public PolynomialModel
    {
    public:
        double b1;  // Coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2
        double b2;  // Coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2
        double b3;  // Coefficients of   y = b1 + b2 * x + b3 * x^2   -OR-   x = b1 + b2 * y + b3 * y^2

    public:

        QuadraticModel(enmIndependentVariable independentVariable)
        {
            _degree = DegreeOfPolynomial::Quadratic;
            MinimumPoints = 3;
            b1 = 0;
            b2 = 0;
            b3 = 0;
            this->independentVariable = independentVariable;
        }

        QuadraticModel(const QuadraticModel& copy)
            : PolynomialModel(copy)
        {
            b1 = copy.b1;
            b2 = copy.b2;
            b3 = copy.b3;
        }

        RegressionModel* Clone() override
        {
            QuadraticModel* newModel = new QuadraticModel(*this);
            return newModel;
        }

        QuadraticModel& operator=(const QuadraticModel& other)
        {
            PolynomialModel::operator=(other);

            b1 = other.b1;
            b2 = other.b2;
            b3 = other.b3;

            return *this;
        }

        float ModeledY(float x) override;

        float ModeledX(float y) override;

    protected:
        class QuadraticSummations : public Summations
        {
        public:
            double x2;
            double xy;
            double x3;
            double x2y;   // (i.e.  SUM(x^2*y))
            double x4;
        };

    public:
        Summations* CalculateSummations(vector<PointF> points) override;

        void CalculateModel(Summations& sums) override;

        void CalculateFeatures() override;
    };

    class QuadraticConsensusModel : public RegressionConsensusModel
    {
    public:
        QuadraticConsensusModel(PolynomialModel::enmIndependentVariable independentVariable) : RegressionConsensusModel()
        {
            model = new QuadraticModel(independentVariable);
            original = new QuadraticModel(independentVariable);

            inliers = vector<PointF>();
            outliers = vector<PointF>();
        }

        QuadraticConsensusModel(const QuadraticConsensusModel& copy)
        {
            model = copy.model;
            original = copy.original;

            inliers = copy.inliers;
            outliers = copy.outliers;
        }

        ~QuadraticConsensusModel()
        {
            delete model;
            delete original;
        }

    protected:
        float CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide) override;
    };

    static QuadraticConsensusModel& CalculateQuadraticRegressionConsensus(vector<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable::X, float sensitivityInPixels = DEFAULT_SENSITIVITY);

public: // Unit tests
    static QuadraticConsensusModel& UnitTestA2(vector<PointF>& anscombe2);
    static QuadraticConsensusModel& UnitTest1(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest2(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest3(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest4(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest5(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest6(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest7(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest8(vector<PointF>& points);
    static QuadraticConsensusModel& UnitTest9(vector<PointF>& points);
};