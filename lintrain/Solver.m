classdef Solver < handle
    %SOLVER Abstract solver for fitting model
    
    methods (Abstract)
        params = calculateParameters(x, y)
        y_hat = applyParameters(x, params)
    end
end
