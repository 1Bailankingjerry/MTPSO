function accuracy = evaluateFeatureSubset(selectedFeatures, X, Y, k)
    if nargin < 4
        k = 5; 
    end
    % 仅使用选定的特征
    X_selected = X(:, selectedFeatures);
    
    % 5折交叉验证
    cv = cvpartition(Y, 'KFold', 5);
    correctPredictions = 0;
    totalPredictions = 0;
    
    for fold = 1:cv.NumTestSets
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        
        model = fitcknn(X_selected(trainIdx, :), Y(trainIdx), 'NumNeighbors', k);
        predictions = predict(model, X_selected(testIdx, :));
        
        % 计算正确的预测数
        correctPredictions = correctPredictions + sum(predictions == Y(testIdx));
        totalPredictions = totalPredictions + length(Y(testIdx));
    end
    
    % 计算正确率
    accuracy = correctPredictions / totalPredictions;
end
