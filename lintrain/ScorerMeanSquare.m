classdef ScorerMeanSquare < Scorer
    %SCORERMEANSQUARE Score fit as mean square error (MSE)
    
    properties
        % a smaller score indicates a better fit
        sort_order = -1
    end
    
    methods
        function s = scorePredictions(~, y, y_hat)
            s = mean((y(:) - y_hat(:)) .^ 2);
        end
    end
end

