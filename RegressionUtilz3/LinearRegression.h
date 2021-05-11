#pragma once
#include <iostream>
#include <string>

#include "PolynomialRegression.h"
#include "RegressionConsensusModel.h"

using namespace std;

/// <summary>
/// LinearRegression
/// Author: Merrill McKee
/// Description:  The purpose of this class is to find the line that best fits through a set of 2D 
///   X, Y points.  It uses Linear Regression or the Least Squares Fit method to do this.  An array
///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
///   find the variables b1 and b2 in the equation y = b1 + b2*x.  The slope of this line is b2.  
///   The y-intercept is b1.  For more information on the formulas used, see the website 
///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression.
///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
///   and ModeledY to get the X value for any given Y position along the parabola or get the Y value for any given
///   X position along the parabola.
///   
///   Note:  Horizontal lines require an independent x-value.  Vertical lines require an independent y-value.
///   
///   Notes: The linear case is a simplication of the matrix math for the quadratic case in the website link:
///          
///          [b2] = [s11]^(-1) * [sY1]
///          
///          [b2] = (1 / s11)  * [sY1]
///          
///          (see the implementation of the quadratic and cubic cases for how this extends to higher degrees)
/// </summary>
class LinearRegression
{
protected:
    const static float DEFAULT_SENSITIVITY;

public:
    class LineModel : public PolynomialModel
    {
    public:
        double slope;
        double intercept;
        double b1;  // Coefficients of   y = b1 + b2 * x   -OR-   x = b1 + b2 * y
        double b2;  // Coefficients of   y = b1 + b2 * x   -OR-   x = b1 + b2 * y

    public:

        LineModel(enmIndependentVariable independentVariable)
        {
            _degree = DegreeOfPolynomial::Linear;
            MinimumPoints = 2;
            slope = 0;
            intercept = 0;
            b1 = 0;
            b2 = 0;
            this->independentVariable = independentVariable;
        }

        LineModel(const LineModel& copy)
            : PolynomialModel(copy)
        {
            slope = copy.slope;
            intercept = copy.intercept;
            b1 = copy.b1;
            b2 = copy.b2;
        }

        RegressionModel* Clone() override
        {
            LineModel* newModel = new LineModel(*this);
            return newModel;
        }

        LineModel& operator=(const LineModel& other)
        {
            PolynomialModel::operator=(other);

            slope = other.slope;
            intercept = other.intercept;
            b1 = other.b1;
            b2 = other.b2;

            return *this;
        }

        float ModeledY(float x) override;

        float ModeledX(float y) override;

    protected:
        class LinearSummations : public Summations
        {
        public:
            double x2;
            double xy;
        };

    public:
        Summations* CalculateSummations(vector<PointF> points) override;

        void CalculateModel(Summations& sums) override;

        void CalculateFeatures() override;
    };

    class LinearConsensusModel : public RegressionConsensusModel
    {
    public:
        LinearConsensusModel(PolynomialModel::enmIndependentVariable independentVariable) : RegressionConsensusModel()
        {
            model = new LineModel(independentVariable);
            original = new LineModel(independentVariable);

            inliers = vector<PointF>();
            outliers = vector<PointF>();
        }

        LinearConsensusModel& operator=(const LinearConsensusModel& other)
        {
            RegressionConsensusModel::operator=(other);
            return *this;
        }

        ~LinearConsensusModel()
        {
            delete model;
            delete original;
        }

    protected:
        float CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide) override;
    };

    static LinearConsensusModel& CalculateLinearRegressionConsensus(vector<PointF> points, PolynomialModel::enmIndependentVariable independentVariable = PolynomialModel::enmIndependentVariable::X, float sensitivityInPixels = DEFAULT_SENSITIVITY);

public: // Unit tests
    static LinearConsensusModel& UnitTestA1(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTestA2(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTestA3(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTestA4(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTest1(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTest2(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTest3(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTest4(vector<PointF>& anscombe1);
    static LinearConsensusModel& UnitTest5(vector<PointF>& anscombe1);

};
