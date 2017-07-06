%% generate random data
num_entries = 50;
num_features = 10;
x = rand(num_entries, num_features);
y = 30 * x(:, 1) - 10 * x(:, 2) + rand(num_entries, 1);

%% create trainer
t = Trainer(x, y, SolverRidgeRegression());
t.debug = 2;

% run
t.runForwardSelection();
%t.runBidirectionalSelection([1, 3]);
