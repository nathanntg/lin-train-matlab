%% generate random data
num_entries = 500;
num_features = 10;
x = rand(num_entries, num_features);
y = (5 * x(:, 1)) + (3 * x(:, 3)) + rand(num_entries, 1);
y = 1 * (y > 3.5);

%% create trainer
s = SolverLogisticRegression();
t = Trainer(x, y, s);
t.debug = 2;

% run
t.runForwardSelection();
%t.runBidirectionalSelection([1, 3]);
