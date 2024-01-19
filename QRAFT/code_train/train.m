data = readtable('../data_processed/inputData.csv');

while true
    data = data(randperm(size(data, 1)), :);
    
    tran1 = data(1:round(height(data)*0.85), 1:18);
    test1 = data(round(height(data)*0.85):height(data), 1:18);
    
    clTrn = unique(tran1{:, 'StateRealProb'});
    clTst = unique(test1{:, 'StateRealProb'});
    
    isSubset = all(ismember(clTst, clTrn));
   
    if isSubset == 1
        break;
    end
end

tran2 = tran1(:, {'CircuitWidth', 'CircuitDepth', 'CircuitNumU1Gates', 'CircuitNumU2Gates', 'CircuitNumU3Gates', 'CircuitNumCXGates', 'StateHammingWeight', 'StateUpProb25', 'StateUpProb50', 'StateUpProb75', 'StateRealProb'});
test2 = test1(:, {'CircuitWidth', 'CircuitDepth', 'CircuitNumU1Gates', 'CircuitNumU2Gates', 'CircuitNumU3Gates', 'CircuitNumCXGates', 'StateHammingWeight', 'StateUpProb25', 'StateUpProb50', 'StateUpProb75', 'StateRealProb'});

tran3 = tran2(:, {'CircuitWidth', 'CircuitDepth', 'CircuitNumU1Gates', 'CircuitNumU2Gates', 'CircuitNumU3Gates', 'CircuitNumCXGates', 'StateHammingWeight', 'StateRealProb'});
test3 = test2(:, {'CircuitWidth', 'CircuitDepth', 'CircuitNumU1Gates', 'CircuitNumU2Gates', 'CircuitNumU3Gates', 'CircuitNumCXGates', 'StateHammingWeight', 'StateRealProb'});

nmcl = length(clTrn);

mscl = zeros(nmcl);
for i = 1:nmcl
    for j = 1:nmcl
        mscl(i, j) = abs(clTrn(i) - clTrn(j));
        mscl(i, j) = ((clTrn(i) - clTrn(j))^2)*(clTrn(i)+1);
    end
end

% Ensemble of trees
mdT1 = fitcensemble(tran1, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));
mdT2 = fitcensemble(tran2, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));
mdT3 = fitcensemble(tran3, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));

save('QRAFT.mat','mdT1');

% kNNs
% mdT1 = fitcknn(tran1, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));
% mdT2 = fitcknn(tran2, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));
% mdT3 = fitcknn(tran3, 'StateRealProb', 'Cost', mscl, 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', 'expected-improvement', 'MaxObjectiveEvaluations', 50));

loss1 = loss(mdT1, test1);
loss2 = loss(mdT2, test2);
loss3 = loss(mdT3, test3);

test1.StatePredPrT1 = predict(mdT1, test1);
test1.StatePredPrT2 = predict(mdT2, test2);
test1.StatePredPrT3 = predict(mdT3, test3);

writetable(test1, '../data_trained/outputData.csv');
