classdef SolverLeastSquares < Solver
    %SOLVERLEASTSQUARES Least squares solver
    %   One of the most common tools used for regression, as it has a nice 
    %   analytic solution that minimizes least square error. This is 
    %   ideally suited to predicting y where y is a function of the 
    %   predictors plus a normal noise term.
    %
    %   Y | X ~ Normal(x, v)
    
    methods
        function params = calculateParameters(~, x, y)
            params = x \ y;
        end
        
        function y_hat = applyParameters(~, x, params)
            y_hat = x * params;
        end
    end
end
