classdef Scorer < handle
    %SCORER Abstract scoring tool for evaluating model fit
    
    properties (Abstract)
        % If a larger score indicates a better fit, then this should return 1.
        % If a smaller score indicates a better fit, then this should return -1.
        sort_order
    end
    
    methods (Abstract)
        s = scorePredictions(y, y_hat)
    end
    
end

