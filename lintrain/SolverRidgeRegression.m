classdef SolverRidgeRegression < Solver
    %SOLVERRIDGEREGRESSION Ridge regression for regularization
    %   Analytically performs ridge regression, where coefficients are 
    %   regularized by learning rate alpha. This constrains coefficients 
    %   and can be effective in situations where over- or under-fitting 
    %   arise. Parameters `alpha` is the regularization constant (alpha of 
    %   0 is least squares, as alpha increases, coefficients are 
    %   increasingly constrained).  Parameter `intercept` (defaults to 
    %   true) causes an intercept column (all ones) to automatically be
    %   detected and excluded from regularization.
    %
    %   Implementation based off of: 
    %   https://gist.github.com/diogojc/1519756
    
    properties
        alpha = 0.1
        intercept = true
    end
    
    methods
        function params = calculateParameters(SL, x, y)
            % make g
            g = SL.alpha * eye(size(x, 2));
            
            % cancel out the intercept terms
            if SL.intercept
                % columns with all 1s
                idx = all(x == 1, 1);
                
                % clear 'em
                g(idx, idx) = 0;
            end
            
            % calculate fit
            params = (x' * x + g' * g) \ (x' * y);
        end
        
        function y_hat = applyParameters(~, x, params)
            y_hat = x * params;
        end
    end
    
end

