#include "EllipticalRegression.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

const float EllipticalRegression::DEFAULT_SENSITIVITY = 0.2f;

float EllipticalRegression::EllipseModel::CalculateRegressionError(PointF point)
{
    return EllipticalRegression::CalculateError(*this, point);
}

RegressionModel::Summations* EllipticalRegression::EllipseModel::CalculateSummations(vector<PointF> points)
{
    EllipseSummations* sum = new EllipseSummations();
    if (points.size() < MinimumPoints)
    {
        sum->N = 0;
        return sum;
    }

    // Initialize all the summations to zero
    sum->x = 0.0;
    sum->y = 0.0;
    sum->x2 = 0.0;
    sum->y2 = 0.0;
    sum->xy = 0.0;
    sum->x3 = 0.0;
    sum->y3 = 0.0;
    sum->x2y = 0.0;  // (i.e.  SUM(x^2*y))
    sum->xy2 = 0.0;
    sum->x4 = 0.0;   // (i.e.  SUM(x^4))
    sum->y4 = 0.0;
    sum->x3y = 0.0;
    sum->x2y2 = 0.0;
    sum->xy3 = 0.0;

    // Shorthand that better matches the math formulas
    auto N = sum->N = (int)points.size();

    // Calculate the summations
    for (auto i = 0; i < N; ++i)
    {
        // Shorthand
        auto x = points[i].X;
        auto y = points[i].Y;
        auto xx = x * x;
        auto xy = x * y;
        auto yy = y * y;

        // Sums
        sum->x += x;
        sum->y += y;
        sum->x2 += xx;
        sum->y2 += yy;
        sum->xy += xy;
        sum->x3 += x * xx;
        sum->y3 += y * yy;
        sum->x2y += xx * y;
        sum->xy2 += x * yy;
        sum->x4 += xx * xx;
        sum->y4 += yy * yy;
        sum->x3y += xx * xy;
        sum->x2y2 += xx * yy;
        sum->xy3 += xy * yy;
    }

    return sum;
}

void EllipticalRegression::EllipseModel::CalculateModel(Summations& sums)
{
    if (sums.N <= 0)
    {
        ValidRegressionModel = false;
        return;
    }

    EllipseSummations& sum = static_cast<EllipseSummations&>(sums);

    // Calculate A = INV(X'X) * X 
    //     or    A = INV(S)   * X

    MatrixXd S(5, 5);
    VectorXd X(5);
    VectorXd A(5);
    
    //Matrix<double> S = Matrix<double>.Build.Dense(5, 5);    // X'X
    //Vector<double> X = Vector<double>.Build.Dense(5);       // X' = [Sx2 Sxy Sy2 Sx Sy]
    //Vector<double> A;// = Vector<double>.Build.Dense(5);       // A  = [a b c d e]

    // Fill the matrices
    X[0] = sum.x2;
    X[1] = sum.xy;
    X[2] = sum.y2;
    X[3] = sum.x;
    X[4] = sum.y;

    // S[column, row]
    S(0, 0) = sum.x4;
    S(1, 0) = S(0, 1) = sum.x3y;
    S(2, 0) = S(1, 1) = S(0, 2) = sum.x2y2;
    S(3, 0) = S(0, 3) = sum.x3;
    S(4, 0) = S(3, 1) = S(1, 3) = S(0, 4) = sum.x2y;
    S(2, 1) = S(1, 2) = sum.xy3;
    S(2, 2) = sum.y4;
    S(4, 1) = S(3, 2) = S(2, 3) = S(1, 4) = sum.xy2;
    S(4, 2) = S(2, 4) = sum.y3;
    S(3, 3) = sum.x2;
    S(4, 3) = S(3, 4) = sum.xy;
    S(4, 4) = sum.y2;

    // Check for a singular matrix
    if (abs(S.determinant()) <= EPSILON)
    {
        ValidRegressionModel = false;
        return;
    }

    A = S.inverse() * X;

    // Calculate the coefficients of ax^2 + bxy + cy^2 + dx + ey + f = 0
    a = A[0];
    b = A[1];
    c = A[2];
    d = A[3];
    e = A[4];
    f = -1.0;

    if (abs(b / a) > EPSILON || abs(b / c) > EPSILON)
    {
        // Tilt angle is not zero
        tilt = 0.5 * atan(b / (c - a)); // todo: move tilt to CalculateFeatures
    }
    else
    {
        tilt = 0.0;
    }

    ValidRegressionModel = true;
}

void EllipticalRegression::EllipseModel::CalculateFeatures()
{
    // Short-hand to match notation in https://www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse
    auto A = this->a;
    auto B = this->b;
    auto C = this->c;
    auto D = this->d;
    auto E = this->e;
    auto c = cos(tilt); // locally redefines "c" to match link notation; careful with "c" vs "C"
    auto s = sin(tilt);

    // Create a model ellipse with the tilt removed
    //    ap*xx + bp*xy + cp*yy + dp*x + ep*y + fp = 0   // where ap denotes a' or "a prime"
    // Substitute x with "cx+sy" and y with "-sx+cy"
    auto cc = c * c;
    auto ss = s * s;
    auto cs = c * s;
    auto Ap = A * cc - B * cs + C * ss;
    auto Bp = 2.0 * A * cs + (cc - ss) * B - 2.0 * C * cs; // zero if the tilt is correctly removed
    auto Cp = A * ss + B * cs + C * cc;
    auto Dp = D * c - E * s;
    auto Ep = D * s + E * c;
    //// Fp = -1.0
    auto Fpp = 1.0 + (Dp * Dp) / (4.0 * Ap) + (Ep * Ep) / (4.0 * Cp);

    if (abs(Bp) > EPSILON)
    {
        //Console.WriteLine("");
        //throw new Exception("CalculateFeatures:  When de-tilting ellipse, the new ellipse does not have a zero coefficient for the xy term");
    }

    if (Ap < 0)
    {
        Ap *= -1.0;
        Cp *= -1.0;
        Dp *= -1.0;
        Ep *= -1.0;
    }

    // Features
    auto x0p = -Dp / (2.0 * Ap) + c * bias.x - s * bias.y; // center of de-tilted ellipse
    auto y0p = -Ep / (2.0 * Cp) + s * bias.x + c * bias.y; // center of de-tilted ellipse
    x0 = (float)(c * x0p + s * y0p);
    y0 = (float)(-s * x0p + c * y0p);
    radiusX = (float)sqrt(abs(Fpp / Ap));
    radiusY = (float)sqrt(abs(Fpp / Cp));
    long_axis = 2.0f * (float)max(radiusX, radiusY);
    short_axis = 2.0f * (float)min(radiusX, radiusY);
}

float EllipticalRegression::EllipseConsensusModel::CalculateError(RegressionModel& modelr, PointF point, bool & pointOnPositiveSide)
{
    EllipseModel& model = static_cast<EllipseModel&>(modelr);

    auto whichSideOfEllipse = WhichSideOfEllipse(model, point);
    if (whichSideOfEllipse == SideOfEllipse::OnPerimeter)
    {
        pointOnPositiveSide = true;
        return 0;
    }
    else if (whichSideOfEllipse == SideOfEllipse::Outside)
    {
        pointOnPositiveSide = true;
    }
    else
    {
        pointOnPositiveSide = false;
    }

    return EllipticalRegression::CalculateError(model, point);
}

// Returns the real roots of the quadratic equation
// root1 > root2
void EllipticalRegression::QuadraticEquation(double a, double b, double c, int& numberOfRoots, float& root1, float& root2)
{
    // -b +/- sqrt( b^2 - 4ac )
    // ------------------------
    //           2a

    root1 = -99999999.9f;
    root2 = -99999999.9f;

    auto discriminant = b * b - 4 * a * c;
    if (discriminant > 0.0)
    {
        numberOfRoots = 2;
        root1 = (float)((-b + sqrt(discriminant)) / (2.0 * a));
        root2 = (float)((-b - sqrt(discriminant)) / (2.0 * a));
    }
    else if (discriminant == 0.0)
    {
        numberOfRoots = 1;
        root1 = (float)(-b / (2.0 * a));
        root2 = root1;
    }
    else
    {
        numberOfRoots = 0;
    }
}

EllipticalRegression::SideOfEllipse EllipticalRegression::WhichSideOfEllipse(EllipseModel ellipse, PointF point)
{
    const double ON_PERIMETER_THRESHOLD = 0.0001;  // A high degree of precision is required to label a point on the perimeter

    auto x = (double)point.X;
    auto y = (double)point.Y;

    auto a = ellipse.a;
    auto b = ellipse.b;
    auto c = ellipse.c;
    auto d = ellipse.d;
    auto e = ellipse.e;
    auto f = ellipse.f;

    auto testMetric = a * x * x +
        b * x * y +
        c * y * y +
        d * x +
        e * y +
        f;

    if (testMetric < -ON_PERIMETER_THRESHOLD)
    {
        return SideOfEllipse::Inside;
    }
    else if (testMetric > ON_PERIMETER_THRESHOLD)
    {
        return SideOfEllipse::Outside;
    }
    else
    {
        return SideOfEllipse::OnPerimeter;
    }
}

// Returns the modeled y-value of an ellipse
float EllipticalRegression::ModeledY(EllipseModel model, float x_orig, EllipseHalves half)
{
    if (!model.ValidRegressionModel)
    {
        // ERROR:  Invalid model
        return -99999999.9f;
    }
    else if (half == EllipseHalves::LeftHalf || half == EllipseHalves::RightHalf)
    {
        // ERROR:  Invalid EllipseHalves parameter
        return -99999999.9f;
    }

    auto x = x_orig - model.bias.x;

    // Short-hand
    auto a = model.a;
    auto b = model.b;
    auto c = model.c;
    auto d = model.d;
    auto e = model.e;
    auto f = model.f;

    int numberOfRoots;
    float root1, root2;
    QuadraticEquation(c, (b * x + e), (a * x * x + d * x + f), numberOfRoots, root1, root2);
    if (numberOfRoots > 0)
    {
        if (half == EllipseHalves::TopHalf)
        {
            return root1 + (float)model.bias.y;
        }
        else
        {
            return root2 + (float)model.bias.y;
        }
    }
    else
    {
        // No y-value for this x-value
        return -99999999.9f;
    }
}

// Returns the modeled x-value of an ellipse
float EllipticalRegression::ModeledX(EllipseModel model, float y_orig, EllipseHalves half)
{
    if (!model.ValidRegressionModel)
    {
        // ERROR:  Invalid model
        return -99999999.9f;
    }
    else if (half == EllipseHalves::TopHalf || half == EllipseHalves::BottomHalf)
    {
        // ERROR:  Invalid EllipseHalves parameter
        return -99999999.9f;
    }

    auto y = y_orig - model.bias.y;

    // Short-hand
    auto a = model.a;
    auto b = model.b;
    auto c = model.c;
    auto d = model.d;
    auto e = model.e;
    auto f = model.f;

    int numberOfRoots;
    float root1, root2;
    QuadraticEquation(a, (b * y + d), (c * y * y + e * y + f), numberOfRoots, root1, root2);
    if (numberOfRoots > 0)
    {
        if (half == EllipseHalves::RightHalf)
        {
            return root1 + (float)model.bias.x;
        }
        else
        {
            return root2 + (float)model.bias.x;
        }
    }
    else
    {
        // No x-value for this y-value
        return -99999999.9f;
    }
}

float EllipticalRegression::CalculateError(RegressionModel& modelr, PointF point)
{
    EllipseModel& model = static_cast<EllipseModel&>(modelr);

    auto x = point.X;
    auto y = point.Y;

    auto errorV1 = abs(EllipticalRegression::ModeledY(model, x, EllipseHalves::TopHalf) - y);
    auto errorV2 = abs(EllipticalRegression::ModeledY(model, x, EllipseHalves::BottomHalf) - y);
    auto errorV = min(errorV1, errorV2);
    auto errorH1 = abs(EllipticalRegression::ModeledX(model, y, EllipseHalves::RightHalf) - x);
    auto errorH2 = abs(EllipticalRegression::ModeledX(model, y, EllipseHalves::LeftHalf) - x);
    auto errorH = min(errorH1, errorH2);
    auto error = min(errorV, errorH);

    if (error >= 0.0f && error < 99999999.9f)
    {
        return error;
    }
    else if (!point.IsEmpty && point.X < 99999999.9f && point.Y < 99999999.9f)
    {
        return max(abs(model.x0 - point.X), abs(model.y0 - point.Y));
    }
    else
    {
        return 99999999.9f;
    }
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::CalculateEllipticalRegressionConsensus(vector<PointF> points, float sensitivity)
{
    auto consensus = new EllipseConsensusModel();
    consensus->Calculate(points, sensitivity);

    return *consensus;
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest1(vector<PointF>& points)
{
    ///////////////////
    // Unit test #1: //
    ///////////////////

    // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
    // [-2 0]
    // [+2 0]
    // [0 -3]
    // [0 +3]
    // [1 sqrt(6.75)]

    points = vector<PointF>();
    points.push_back(PointF(-2.0f, 0.0f));
    points.push_back(PointF(2.0f, 0.0f));
    points.push_back(PointF(0.0f, -3.0f));
    points.push_back(PointF(0.0f, 3.0f));
    points.push_back(PointF(1.0f, (float)sqrt(6.75)));
    points.push_back(PointF(1.0f, -(float)sqrt(6.75)));

    return CalculateEllipticalRegressionConsensus(points);
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest2(vector<PointF>& points)
{
    ///////////////////
    // Unit test #1b: //
    ///////////////////

    // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
    // [-2 0]
    // [+2 0]
    // [0 -3]
    // [0 +3]
    // [1 sqrt(6.75)]

    points = vector<PointF>();
    points.push_back(PointF(498.0f, 400.0f));
    points.push_back(PointF(502.0f, 400.0f));
    points.push_back(PointF(500.0f, 397.0f));
    points.push_back(PointF(500.0f, 403.0f));
    points.push_back(PointF(501.0f, 400.0f + (float)sqrt(6.75)));
    points.push_back(PointF(501.0f, 400.0f - (float)sqrt(6.75)));

    return CalculateEllipticalRegressionConsensus(points);
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest3(vector<PointF>& points)
{
    ///////////////////
    // Unit test #2: //
    ///////////////////

    // A off-center ellipse with 30 degree tilt:
    //              Center: (2,1)
    //                   a: 5
    //                   b: 3
    //                Tilt: 30 degrees counter-clockwise
    // [6.33 3.5]
    // [5.0  4.46]
    // [2.87 4.5]
    // [0.5  3.60]
    // [-1.46 2.0]
    // [-2.5  0.134]
    // [-2.33 -1.5]

    points = vector<PointF>();
    points.push_back(PointF(6.33013f, 3.5f));
    points.push_back(PointF(5.0f, 4.46410f));
    points.push_back(PointF(2.86603f, 4.5f));
    points.push_back(PointF(0.5f, 3.59808f));
    points.push_back(PointF(-1.46410f, 2.0f));
    points.push_back(PointF(-2.5f, 0.13397f));
    points.push_back(PointF(-2.33013f, -1.5f));

    return CalculateEllipticalRegressionConsensus(points);
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest4(vector<PointF>& points)
{
    ///////////////////
    // Unit test #3: //
    ///////////////////

    // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
    // [-2 0]
    // [+2 0]
    // [0 -3]
    // [0 +3]
    // [1 sqrt(6.75)]

    points = vector<PointF>();
    points.push_back(PointF(-2.0f, 0.0f));
    points.push_back(PointF(2.0f, 0.0f));
    points.push_back(PointF(0.0f, -3.0f));
    points.push_back(PointF(0.0f, 3.0f));
    points.push_back(PointF(1.0f, (float)sqrt(6.75)));
    points.push_back(PointF(1.0f, -(float)sqrt(6.75)));
    points.push_back(PointF(-1.0f, (float)sqrt(6.75)));
    points.push_back(PointF(-1.0f, -(float)sqrt(6.75)));
    points.push_back(PointF(0.0f, 1.0f)); // internal outlier

    return CalculateEllipticalRegressionConsensus(points);
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest5(vector<PointF>& points)
{
    ///////////////////
    // Unit test #3: //
    ///////////////////

    // A centered ellipse using the equation:     x^2/4 + y^2/9 = 1
    // [-2 0]
    // [+2 0]
    // [0 -3]
    // [0 +3]
    // [1 sqrt(6.75)]

    points = vector<PointF>();
    points.push_back(PointF(-2.0f, 0.0f));
    points.push_back(PointF(2.0f, 0.0f));
    points.push_back(PointF(0.0f, -3.0f));
    points.push_back(PointF(0.0f, 3.0f));
    points.push_back(PointF(1.0f, (float)sqrt(6.75)));
    points.push_back(PointF(1.0f, -(float)sqrt(6.75)));
    points.push_back(PointF(-1.0f, (float)sqrt(6.75)));
    points.push_back(PointF(-1.0f, -(float)sqrt(6.75)));
    points.push_back(PointF(4.0f, 4.0f));
    points.push_back(PointF(3.0f, 3.0f));
    points.push_back(PointF(2.5f, 2.5f));

    return CalculateEllipticalRegressionConsensus(points);
}

EllipticalRegression::EllipseConsensusModel& EllipticalRegression::UnitTest6(vector<PointF>& points)
{
    ///////////////////
    // Unit test #4: //
    ///////////////////

    // A off-center ellipse with 30 degree tilt:
    //              Center: (2,1)
    //                   a: 5
    //                   b: 3
    //                Tilt: 30 degrees counter-clockwise
    // [6.33 3.5]
    // [5.0  4.46]
    // [2.87 4.5]
    // [0.5  3.60]
    // [-1.46 2.0]
    // [-2.5  0.134]
    // [-2.33 -1.5]

    points = vector<PointF>();
    points.push_back(PointF(4.0f, -7.0f));
    points.push_back(PointF(4.0f, -4.0f));
    points.push_back(PointF(6.33013f, 3.5f));
    points.push_back(PointF(5.0f, 4.46410f));
    points.push_back(PointF(2.86603f, 4.5f));
    points.push_back(PointF(0.5f, 3.59808f));
    points.push_back(PointF(-1.46410f, 2.0f));
    points.push_back(PointF(-2.5f, 0.13397f));
    points.push_back(PointF(-2.33013f, -1.5f));
    points.push_back(PointF(5.8f, 0.45f));
    points.push_back(PointF(0.63f, -2.5f));

    return CalculateEllipticalRegressionConsensus(points);
}