classdef Trainer < handle
    %TRAINER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % training data
        x
        y
        
        % k-fold
        folds_count
        folds_seed = 37444887; % consistent k-fold
        
        % fitting tools
        solver
        scorer
        
        % post-training data
        column_indices
        fit
        score
        
        % debug
        debug = 0
    end
    
    properties (Access=protected)
        folds
    end
    
    methods
        function TR = Trainer(x, y, solver, scorer, folds_count)
            TR.x = x;
            TR.y = y;
            
            if ~exist('solver', 'var') || isempty(solver)
                TR.solver = SolverLeastSquares();
            else
                TR.solver = solver;
            end
            
            if ~exist('scorer', 'var') || isempty(scorer)
                TR.scorer = ScorerMeanAbsolute();
            else
                TR.scorer = scorer;
            end
            
            if ~exist('folds_count', 'var') || isempty(folds_count)
                TR.folds_count = 5;
            else
                TR.folds_count = folds_count;
            end
        end
        
        function runForwardSelection(TR, initial_columns, initial_score)
            if ~exist('initial_columns', 'var')
                initial_columns = [];
            end
            if ~exist('initial_score', 'var')
                initial_score = [];
            end
            
            % configure
            TR.cutFolds();
            TR.configureInitialConditions(initial_columns, initial_score, false);
            
            % run forward
            TR.runFeatureSelection(true, false);
        end
        
        function runBidirectionalSelection(TR, initial_columns, initial_score)
            if ~exist('initial_columns', 'var')
                initial_columns = [];
            end
            if ~exist('initial_score', 'var')
                initial_score = [];
            end
            
            % configure
            TR.cutFolds();
            TR.configureInitialConditions(initial_columns, initial_score, false);
            
            % run bidrectional
            TR.runFeatureSelection(true, true);
        end
        
        function runBackwardSelection(TR, initial_columns, initial_score)
            if ~exist('initial_columns', 'var')
                initial_columns = [];
            end
            if ~exist('initial_score', 'var')
                initial_score = [];
            end
            
            % configure
            TR.cutFolds();
            TR.configureInitialConditions(initial_columns, initial_score, true);
            
            % run backward
            TR.runFeatureSelection(false, true);
        end
        
        function x_sub = selectColumns(TR, x)
            if ismatrix(x) && all(size(x) > 1)
                x_sub = x(:, TR.column_indices);
            else
                x_sub = x(TR.column_indices);
            end
        end
        
        function y_hat = applyModel(TR, x)
            if ismatrix(x) && all(size(x) > 1)
                y_hat = TR.solver.applyParameters(x(:, TR.column_indices), TR.fit);
            else
                y_hat = TR.solver.applyParameters(x(TR.column_indices), TR.fit);
            end
        end
    end
    
    methods (Access=protected)
        function cutFolds(TR)
            % seed 
            old_rng = rng;
            if ~isempty(TR.folds_seed)
                rng(TR.folds_seed);
            end
            
            % indices
            indices_count = size(TR.x, 1);
            indices = randperm(indices_count);
            
            % reset seed
            if ~isempty(TR.folds_seed)
                rng(old_rng);
            end
            
            % number per fold
            per_fold_count = ceil(indices_count / TR.folds_count);
            
            % make folds
            TR.folds = {};
            for i = 1:TR.folds_count
                idx_start = 1 + (i - 1) * per_fold_count;
                idx_end = max(i * per_fold_count, indices_count);
                TR.folds{i} = indices(idx_start:idx_end);
            end
        end
        
        function [training, validation] = getFold(TR, fold)
            training = TR.folds{fold};
            validation = cat(2, TR.folds{(1:TR.folds_count) ~= fold});
        end
        
        function params = train(TR, col_indices, row_indices)
            % get x and y
            cur_x = TR.x(row_indices, col_indices);
            cur_y = TR.y(row_indices);
            
            % fit model
            params = TR.solver.calculateParameters(cur_x, cur_y);
        end
        
        function score = validate(TR, col_indices, row_indices, params)
            % get x and y
            cur_x = TR.x(row_indices, col_indices);
            cur_y = TR.y(row_indices);
            
            % generate predictions
            cur_y_hat = TR.solver.applyParameters(cur_x, params);
            
            % score
            score = TR.scorer.scorePredictions(cur_y, cur_y_hat);
        end
        
        function score = scoreColumns(TR, col_indices)
            score = 0;
            
            for i = 1:TR.folds_count
                % get row indices
                [rows_for_training, rows_for_validation] = TR.getFold(i);
                
                % train and fit
                params = TR.train(col_indices, rows_for_training);
                
                % validation score
                score = score + TR.validate(col_indices, rows_for_validation, params);
            end
            
            % normalize
            score = score / TR.folds_count;
        end
        
        function l = isBetterScore(TR, best_score, score)
            l = isempty(best_score) || ((score - best_score) * TR.scorer.sort_order) > 0;
        end
        
        function configureInitialConditions(TR, initial_column_indices, initial_score, backward)
            % clear scofre
            TR.score = [];
            
            % allow starting with initial column selection
            if isempty(initial_column_indices)
                if backward
                    % start with all columns
                    TR.column_indices = 1:size(TR.x, 2);
                    TR.score = TR.scoreColumns(TR.column_indices);
                else
                    TR.column_indices = [];
                end
            else
                TR.column_indices = initial_column_indices;
                if isempty(initial_score)
                    TR.score = TR.scoreColumns(TR.column_indices);
                else
                    TR.score = initial_score;
                end
            end
        end
        
        function [new_col_indices, best_score] = doForwardSelection(TR, col_indices, best_score)
            % potential column indices
            num_col_indices = size(TR.x, 2);
            
            % empty vector means nothing to add
            new_col_indices = [];
            
            % for each potential column index
            for potential_col_index = setdiff(1:num_col_indices, col_indices)
                % full list
                potential_col_indices = [col_indices potential_col_index];
                
                % score it
                cur_score = TR.scoreColumns(potential_col_indices);
                
                % is better score?
                if TR.isBetterScore(best_score, cur_score)
                    best_score = cur_score;
                    new_col_indices = potential_col_indices;
                end
            end
        end
        
        function [new_col_indices, best_score] = doBackwardSelection(TR, col_indices, best_score)
            % empty vector means nothing to add
            new_col_indices = [];
            
            % for each potential column index
            for potential_col_index = col_indices
                % full list
                potential_col_indices = setdiff(col_indices, potential_col_index);
                
                % score it
                cur_score = TR.scoreColumns(potential_col_indices);
                
                % is better score?
                if TR.isBetterScore(best_score, cur_score)
                    best_score = cur_score;
                    new_col_indices = potential_col_indices;
                end
            end
        end
        
        function runFeatureSelection(TR, forward, backward)
            % initial values
            col_indices = TR.column_indices;
            cur_score = TR.score;
            
            % training loop
            while true
                % if running forward selection or bidirectional 
                % selection...
                if forward
                    % try adding a column
                    [new_col_indices, cur_score] = TR.doForwardSelection(col_indices, cur_score);
                    
                    % success
                    if ~isempty(new_col_indices)
                        % debugging
                        if TR.debug >= 2
                            fprintf('Added column %d (score: %f)\n', new_col_indices(end), cur_score);
                        end
                        
                        col_indices = new_col_indices;
                        continue;
                    end
                end
                
                % if running backwards selection or bidirectional
                % selection...
                if backward
                    % try removing a column
                    [new_col_indices, cur_score] = TR.doBackwardSelection(col_indices, cur_score);
                    
                    % success
                    if ~isempty(new_col_indices)
                        % debugging
                        if TR.debug >= 2
                            fprintf('Removed column %d (score: %f)\n', setdiff(col_indices, new_col_indices), cur_score);
                        end
                        
                        col_indices = new_col_indices;
                        continue;
                    end
                end
                
                % do not iterate
                break;
            end
            
            % set score
            TR.score = cur_score;
            
            % set fit and column indices
            TR.column_indices = col_indices;
            TR.fit = TR.train(col_indices, 1:size(TR.x, 1));
            
            % print debugging information
            if TR.debug >= 1
                fprintf('Final score: %f\n', TR.score);
            end
            if TR.debug >= 2
                fprintf('Columns: %s\n', sprintf('%d ', TR.column_indices));
                fprintf('Fit: %s\n', sprintf('%d ', TR.fit));
            end
        end
    end
end

