This repository contains statistical regression utilities written in C++ in Visual Studio 2019.  The utilities include standard least-squares linear, quadratic, cubic, and elliptical regression.  In addition to standard least squares regression, they contain "consensus regression" that automatically detects outliers in a dataset, removes them from the inliers, and updates the least-squares regression model quickly.

The image below is an example of linear regression and linear regression consensus.  The red line is a standard least squares linear regression on all blue data points.  The consensus algorithm identifies outliers (red stars) and then adjusts the regression model to ignore these outliers.

![image](https://user-images.githubusercontent.com/79757625/117740886-5fe32380-b1cf-11eb-8076-8a2fe13c46f1.png)

Here is an example that may apply to edge points detected in an image at a corner.  The original regression (red) fits a line through both lines.  The consensus algorithm identifies outliers (red) and fits its model to the dominant line.
![image](https://user-images.githubusercontent.com/79757625/117740905-67a2c800-b1cf-11eb-8613-0927d241d1c5.png)

Anscombe's quartet is four sets of data that have virtually the same linear regression function.  In 3 of the 4 cases, outliers have a noticeable impact.  Two of the cases, a single outlier is enough to make the linear regression of poorer to unusable quality.  This quartet helps demonstate the sensitivity of common regression tools to outliers.  For more reading see https://en.wikipedia.org/wiki/Anscombe%27s_quartet.

The following images illustrate linear regression on the Anscombe's quartet:

![image](https://user-images.githubusercontent.com/79757625/117516460-260fe400-af67-11eb-94b9-02d05308799f.png)
![image](https://user-images.githubusercontent.com/79757625/117740919-712c3000-b1cf-11eb-8465-311aa3ba1d53.png)
![image](https://user-images.githubusercontent.com/79757625/117741070-8903b400-b1cf-11eb-9f56-e81d55762edd.png)
![image](https://user-images.githubusercontent.com/79757625/117741031-7ab59800-b1cf-11eb-94d4-f7c09c72af83.png)
![image](https://user-images.githubusercontent.com/79757625/117741054-81dca600-b1cf-11eb-8910-0f70db07fbdf.png)

Some examples of quadratic regression and quadratic regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117741082-902ac200-b1cf-11eb-8779-47d79c3a9289.png)
![image](https://user-images.githubusercontent.com/79757625/117741095-96b93980-b1cf-11eb-8e40-01313ed83f6e.png)
![image](https://user-images.githubusercontent.com/79757625/117741103-9caf1a80-b1cf-11eb-9d13-983923b835de.png)
![image](https://user-images.githubusercontent.com/79757625/117741111-a2a4fb80-b1cf-11eb-88f6-4933c505d3e7.png)
![image](https://user-images.githubusercontent.com/79757625/117741124-a89adc80-b1cf-11eb-9178-e955f99c9753.png)
![image](https://user-images.githubusercontent.com/79757625/117741132-ae90bd80-b1cf-11eb-9b84-c6ff0668a6f2.png)
![image](https://user-images.githubusercontent.com/79757625/117741139-b3557180-b1cf-11eb-8e65-c4035892ab92.png)
![image](https://user-images.githubusercontent.com/79757625/117741145-b8b2bc00-b1cf-11eb-9af3-40bdb37868f1.png)

Some examples of cubic regression and cubic regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117741155-bfd9ca00-b1cf-11eb-9350-094a88db9f28.png)
![image](https://user-images.githubusercontent.com/79757625/117741162-c5371480-b1cf-11eb-995a-1f9c8e5e6e7a.png)
![image](https://user-images.githubusercontent.com/79757625/117741170-cb2cf580-b1cf-11eb-8cac-8bc9b6d135f5.png)
![image](https://user-images.githubusercontent.com/79757625/117741180-d08a4000-b1cf-11eb-9aea-6d10d915d766.png)

Some examples of elliptical regression and elliptical regression consensus:
![image](https://user-images.githubusercontent.com/79757625/117741193-d6802100-b1cf-11eb-8552-1a340e2277d6.png)
![image](https://user-images.githubusercontent.com/79757625/117741203-dbdd6b80-b1cf-11eb-9b02-b7e85baff3d3.png)
![image](https://user-images.githubusercontent.com/79757625/117741215-e26be300-b1cf-11eb-97c8-8948bf5bf5c6.png)
![image](https://user-images.githubusercontent.com/79757625/117741225-e7c92d80-b1cf-11eb-963c-94d1dc88f6fb.png)
![image](https://user-images.githubusercontent.com/79757625/117741233-ec8de180-b1cf-11eb-8b5e-aef441d0fa0c.png)

On the algorithm.  The algorithm takes some inspiration from RANSAC but where RANSAC builds random models from the smallest subsamples, this algorithm starts with all datapoints and iteratively labels and removes outliers.  Instead of an exhaustive search for the worst outlier to remove, only 3 candidates are considered.  Two of the candidates are those candidates with the greatest regression error "above" and "below" the model (substitute "left"/"right"/"inside"/"outside").  Since the summations were kept for the least squares calculations, removing an outlier is generally as simple as decrementing these summations.  Originally, only two candidates were going to be considered but in testing this initial algorithm it was not removing outliers for some "easy" cases.  I added a third candidate which is the candidate with maximum "influence"; this is generally the data point with maximum x * y.  Finally, iteration is stopped when the model's average regression error goes below a threshold.

For display, matplotlib-cpp (matplotlib for C++) was used.

A tree hierarchy of the classes  
- RegressionModel  
  - PolynomialModel  
    - LinearModel  
    - QuadraticModel  
    - CubicModel  
  - EllipticalModel  

There is a parallel C# implementation of these utilities at https://github.com/merrillmckee/regressionUtilsCSharp


