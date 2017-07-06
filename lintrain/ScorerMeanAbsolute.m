classdef ScorerMeanAbsolute < Scorer
    %SCORERMEANABSOLUTE Score fit as mean absolute error (MAE)
    
    properties
        % a smaller score indicates a better fit
        sort_order = -1
    end
    
    methods
        function s = scorePredictions(~, y, y_hat)
            s = mean(abs(y(:) - y_hat(:)));
        end
    end
end

