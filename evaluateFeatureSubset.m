function accuracy = evaluateFeatureSubset(selectedFeatures, X_train, Y_train, X_test, Y_test, k)
    % 仅使用选定的特征
    X_train_selected = X_train(:, selectedFeatures);
    X_test_selected = X_test(:, selectedFeatures);
    
    % 训练 KNN 分类器
    model = fitcknn(X_train_selected, Y_train, 'NumNeighbors', k);
    
    % 在测试集上进行分类预测
    predictions = predict(model, X_test_selected);
    
    % 计算正确的预测数
    correctPredictions = sum(predictions == Y_test);
    totalPredictions = length(Y_test);
    
    % 计算正确率
    accuracy = correctPredictions / totalPredictions;
end
