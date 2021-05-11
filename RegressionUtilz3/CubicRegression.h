#pragma once
#include <iostream>
#include <string>

#include "PolynomialRegression.h"
#include "RegressionConsensusModel.h"

using namespace std;

/// <summary>
/// CubicRegression
/// Author: Merrill McKee
/// Description:  The purpose of this class is to find the curve that best fits through a set of 2D 
///   X, Y points.  It uses Cubic Regression or the Least Squares Fit method to do this.  An array
///   list of System.Drawing.PointF objects are passed in to the constructor.  These points are used to 
///   find the variables b1, b2, b3, and b4 in the equation y = b1 + b2*x + b3*x^2 + b4*x^3.  For more 
///   information on the formulas used, see the website 
///   http://math.stackexchange.com/questions/267865/equations-for-quadratic-regression and the notes 
///   below.  Also, look at my notes for the CubicRegression.cs implementation and use 
///   http://www.dr-lex.be/random/matrix-inv.html for the explicit equations for a 3x3 inverse.
///   
///   Once these variables have been calculated in the constructor then the user of the class can call ModeledX
///   and ModeledY to get the X value for any given Y position along the curve or get the Y value for any given
///   X position along the curve.
///   
///   Notes: Additional matrix algebra details not in the website link:
///          
///          [b2] = [s11 s12 s13]-1 * [sY1]
///          [b3]   [s12 s22 s23]     [sY2]
///          [b4]   [s13 s23 s33]     [sY3]
///          
///          [b2] = (1 / det(S)) [ (s22s33-s23s23)  (s13s23-s12s33)  (s12s23-s13s22) ] * [sY1]
///          [b3]                [ (s13s23-s12s33)  (s11s33-s13s13)  (s12s13-s11s23) ]   [sY2]
///          [b4]                [ (s12s23-s13s22)  (s12s13-s11s23)  (s11s22-s12s12) ]   [sY3]
///          
///             where  det(S) = s11(s22s33-s23s23) - s12(s12s33-s13s23) + s13(s12s23-s13s22)
///          
///             using t11 = (s22s33-s23s23)       OR       [t11 t12 t13]
///                   t12 = (s13s23-s12s33)                [t12 t22 t23]
///                   t13 = (s12s23-s13s22)                [t13 t23 t33]
///                   t22 = (s11s33-s13s13)
///                   t23 = (s12s13-s11s23)
///                   t33 = (s11s22-s12s12)
/// 
///          [b2] = (1 / det(S)) [t11 t12 t13] * [sY1]
///          [b3]                [t12 t22 t23]   [sY2]
///          [b4]                [t13 t23 t33]   [sY3]
///          
/// </summary>

class CubicRegression : public PolynomialModel
{
protected:
    const static float DEFAULT_SENSITIVITY;

public:
    class CubicModel : public PolynomialModel
    {
    public:
        double b1;  // Coefficients of   y = b1 + b2 * x + b3 * x^2 + b4 * x^3  -OR-   x = b1 + b2 * y + b3 * y^2 + b4 * y^3
        double b2;  // Coefficients of   y = b1 + b2 * x + b3 * x^2 + b4 * x^3  -OR-   x = b1 + b2 * y + b3 * y^2 + b4 * y^3
        double b3;  // Coefficients of   y = b1 + b2 * x + b3 * x^2 + b4 * x^3  -OR-   x = b1 + b2 * y + b3 * y^2 + b4 * y^3
        double b4;  // Coefficients of   y = b1 + b2 * x + b3 * x^2 + b4 * x^3  -OR-   x = b1 + b2 * y + b3 * y^2 + b4 * y^3

    public:

        CubicModel(enmIndependentVariable independentVariable)
        {
            _degree = DegreeOfPolynomial::Cubic;
            MinimumPoints = 4;
            b1 = 0;
            b2 = 0;
            b3 = 0;
            b4 = 0;
            this->independentVariable = independentVariable;
        }

        CubicModel(const CubicModel& copy)
            : PolynomialModel(copy)
        {
            b1 = copy.b1;
            b2 = copy.b2;
            b3 = copy.b3;
            b4 = copy.b4;
        }

        RegressionModel* Clone() override
        {
            CubicModel* newModel = new CubicModel(*this);
            return newModel;
        }

        CubicModel& operator=(const CubicModel& other)
        {
            PolynomialModel::operator=(other);

            b1 = other.b1;
            b2 = other.b2;
            b3 = other.b3;
            b4 = other.b4;

            return *this;
        }

        float ModeledY(float x) override;

        float ModeledX(float y) override;

    protected:
        class CubicSummations : public Summations
        {
        public:
            double x2;
            double x3;
            double x4;
            double x5;
            double x6;
            double xy;
            double x2y;   // (i.e.  SUM(x^2*y))
            double x3y;
        };

    public:
        Summations* CalculateSummations(vector<PointF> points) override;

        void CalculateModel(Summations& sums) override;

        void CalculateFeatures() override;
    };

    class CubicConsensusModel : public RegressionConsensusModel
    {
    public:
        CubicConsensusModel(PolynomialModel::enmIndependentVariable independentVariable) : RegressionConsensusModel()
        {
            model = new CubicModel(independentVariable);
            original = new CubicModel(independentVariable);

            inliers = vector<PointF>();
            outliers = vector<PointF>();
        }

        CubicConsensusModel(const CubicConsensusModel& copy)
        {
            model = copy.model;
            original = copy.original;

            inliers = copy.inliers;
            outliers = copy.outliers;
        }

        ~CubicConsensusModel()
        {
            delete model;
            delete original;
        }

    protected:
        float CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide) override;
    };

    static CubicConsensusModel& CalculateCubicRegressionConsensus(vector<PointF> points, enmIndependentVariable independentVariable = enmIndependentVariable::X, float sensitivityInPixels = DEFAULT_SENSITIVITY);

public: // Unit tests
    static CubicConsensusModel& UnitTest1(vector<PointF>& points);
    static CubicConsensusModel& UnitTest2(vector<PointF>& points);
    static CubicConsensusModel& UnitTest3(vector<PointF>& points);
    static CubicConsensusModel& UnitTest4(vector<PointF>& points);
    static CubicConsensusModel& UnitTest5(vector<PointF>& points);
    static CubicConsensusModel& UnitTest6(vector<PointF>& points);
    static CubicConsensusModel& UnitTest7(vector<PointF>& points);
    static CubicConsensusModel& UnitTest8(vector<PointF>& points);
    static CubicConsensusModel& UnitTest9(vector<PointF>& points);
    static CubicConsensusModel& UnitTest10(vector<PointF>& points);
    static CubicConsensusModel& UnitTest11(vector<PointF>& points);
    static CubicConsensusModel& UnitTest12(vector<PointF>& points);
};