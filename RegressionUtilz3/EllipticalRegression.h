#pragma once
#include <iostream>
#include <string>

#include "RegressionModel.h"
#include "RegressionConsensusModel.h"

using namespace std;

/// <summary>
/// EllipticalRegression
/// Author: Merrill McKee
/// Description:  The purpose of this class is to find the ellipse/hyperbola that best fits through a set of 2D 
///   X, Y points.  It uses the Least Squares Fit method to do this.  A list of System.Drawing.PointF 
///   objects are passed in to the constructor.  These points are used to find the variables a, b, c, d, and e 
///   in the equation.  f is assumed to be -1.
///         ax^2 + bxy + cy^2 + dx + ey + f = 0
///   For more information on the algorithm/formulas used, see the website 
///   http://www.mathworks.com/matlabcentral/fileexchange/3215-fit-ellipse/content/fit_ellipse.m and 
///   https://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf and the notes 
///   below.  The method is slightly different than the existing algorithms used in the linear, quadratic, 
///   and cubic regressions implemented in this same project.  Those use explicit derivations where this 
///   method requires calculating the inverse of a 5x5 matrix.
///   
///   Notes: Additional matrix algebra details not in the website link:
///   Note:  Using the following shorthand:  Sxy == SUM(xy)  == SIGMA(xy)  over all data values
///                                          Sx2 == SUM(x^2) == SIGMA(x^2) over all data values
///          
///           A  = INV(X'X) * X
///          
///          [a] = [Sx4   Sx3y  Sx2y2 Sx3   Sx2y]-1 * [Sx2]
///          [b]   [Sx3y  Sx2y2 Sxy3  Sx2y  Sxy2]     [Sxy]
///          [c]   [Sx2y2 Sxy3  Sx2y2 Sxy2  Sy3 ]     [Sy2]
///          [d]   [Sx3   Sx2y  Sxy2  Sx2   Sxy ]     [Sx ]
///          [e]   [Sx2y  Sxy2  Sy3   Sxy   Sy2 ]     [Sy ]
/// 
/// </summary>

class EllipticalRegression : public RegressionModel
{
protected:
    const static float DEFAULT_SENSITIVITY;

    enum class SideOfEllipse
    {
        Inside = -1,
        OnPerimeter = 0,
        Outside = 1
    };

public:
    enum class EllipseHalves
    {
        TopHalf = 1,
        BottomHalf = 2,
        RightHalf = 3,
        LeftHalf = 4
    };

    class EllipseModel : public RegressionModel
    {
    public:
        double a;
        double b;
        double c;
        double d;
        double e;
        double f;
        float x0;
        float y0;
        double tilt;
        float radiusX;
        float radiusY;
        float long_axis;
        float short_axis;

    public:
        EllipseModel()
        {
            MinimumPoints = 5;
            a = 0;
            b = 0;
            c = 0;
            d = 0;
            e = 0;
            f = 0;
            x0 = 0;
            y0 = 0;
            tilt = 0;
            radiusX = 0;
            radiusY = 0;
            long_axis = 0;
            short_axis = 0;
        }

        EllipseModel(const EllipseModel& copy)
            : RegressionModel(copy)
        {
            MinimumPoints = copy.MinimumPoints;
            a = copy.a;
            b = copy.b;
            c = copy.c;
            d = copy.d;
            e = copy.e;
            f = copy.f;
            x0 = copy.x0;
            y0 = copy.y0;
            tilt = copy.tilt;
            radiusX = copy.radiusX;
            radiusY = copy.radiusY;
            long_axis = copy.long_axis;
            short_axis = copy.short_axis;
        }

        RegressionModel* Clone() override
        {
            EllipseModel* newModel = new EllipseModel(*this);
            return newModel;
        }

        EllipseModel& operator=(const EllipseModel& other)
        {
            RegressionModel::operator=(other);

            MinimumPoints = other.MinimumPoints;
            a = other.a;
            b = other.b;
            c = other.c;
            d = other.d;
            e = other.e;
            f = other.f;
            x0 = other.x0;
            y0 = other.y0;
            tilt = other.tilt;
            radiusX = other.radiusX;
            radiusY = other.radiusY;
            long_axis = other.long_axis;
            short_axis = other.short_axis;

            return *this;
        }

    protected:
        class EllipseSummations : public Summations
        {
        public:
            double x2;
            double y2;
            double xy;
            double x3;
            double y3;
            double x2y;   // (i.e.  SUM(x^2*y))
            double xy2;
            double x4;     // (i.e.  SUM(x^4))
            double y4;
            double x3y;
            double x2y2;
            double xy3;
        };
        
    public:
        float CalculateRegressionError(PointF point) override;

        Summations* CalculateSummations(vector<PointF> points) override;

        void CalculateModel(Summations& sums) override;

        void CalculateFeatures() override;
    };

    class EllipseConsensusModel : public RegressionConsensusModel
    {
    public:
        EllipseConsensusModel() : RegressionConsensusModel()
        {
            influenceError = InfluenceError::L2;
            model = new EllipseModel();
            original = new EllipseModel();

            inliers = vector<PointF>();
            outliers = vector<PointF>();
        }

        EllipseConsensusModel(const EllipseConsensusModel& copy)
        {
            influenceError = copy.influenceError;
            model = copy.model;
            original = copy.original;

            inliers = copy.inliers;
            outliers = copy.outliers;
        }

        ~EllipseConsensusModel()
        {
            delete model;
            delete original;
        }

    protected:
        float CalculateError(RegressionModel& model, PointF point, bool& pointOnPositiveSide) override;
    };

    static EllipseConsensusModel& CalculateEllipticalRegressionConsensus(vector<PointF> points, float sensitivity = DEFAULT_SENSITIVITY);
    static void QuadraticEquation(double a, double b, double c, int& numberOfRoots, float& root1, float& root2);
    static SideOfEllipse WhichSideOfEllipse(EllipseModel ellipse, PointF point);
    static float ModeledY(EllipseModel model, float x_orig, EllipseHalves half = EllipseHalves::TopHalf);
    static float ModeledX(EllipseModel model, float y_orig, EllipseHalves half = EllipseHalves::RightHalf);
    static float CalculateError(RegressionModel& model, PointF point);

public: // Unit tests
    static EllipseConsensusModel& UnitTest1(vector<PointF>& points);
    static EllipseConsensusModel& UnitTest2(vector<PointF>& points);
    static EllipseConsensusModel& UnitTest3(vector<PointF>& points);
    static EllipseConsensusModel& UnitTest4(vector<PointF>& points);
    static EllipseConsensusModel& UnitTest5(vector<PointF>& points);
    static EllipseConsensusModel& UnitTest6(vector<PointF>& points);
};
