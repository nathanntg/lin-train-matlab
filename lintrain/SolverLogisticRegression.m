classdef SolverLogisticRegression < Solver
    %SOLVERLOGISTICREGRESSION Logistic regression
    %   This uses the generalized linear model with gradient descent to 
    %   calculate linear coefficients (parameters). This class is abstract 
    %   and must be extended by the underlying distribution to determine 
    %   the exact gradient and method for applying parameters.
    % 
    %   Modeled after the code:
    %   http://www.cs.cmu.edu/~ggordon/IRLS-example/
    
    properties
        epsilon = 1e-10;
        max_iterations = 500;
        ridge = 1e-5;
    end
    
    methods
        function params = calculateParameters(SL, x, y)
            % dimensions
            n = size(x, 1); % rows
            m = size(x, 2); % columns
            
            params = zeros(m, 1);
            old_exp_y = - ones(size(y));
            
            % convert to matrix
            ridgemat = SL.ridge * eye(m);
            
            for i = 1:SL.max_iterations
                % calculate
                adj_y = x * params;
                exp_y = 1 ./ (1 + exp(-adj_y));
                deriv = exp_y .* (1 - exp_y);
                w_adj_y = (deriv .* adj_y + (y - exp_y));
                weights = diag(deriv(:));
                
                % update
                params = (x' * weights * x + ridgemat) \ x' * w_adj_y;
                
                % compare with epsilon
                if sum(abs(exp_y - old_exp_y)) < n * SL.epsilon
                    return;
                end
                
                old_exp_y = exp_y;
            end
            
            warning('Did not converge.');
        end
        
        function y_hat = applyParameters(~, x, params)
            y_hat = 1 ./ (1 + exp(- x * params));
        end
    end
    
end

