function fitness = calculateFitness(selectedFeatures, X, Y, alpha)
  
    numClasses = numel(unique(Y));
    numSelected = sum(selectedFeatures);
    numAll = length(selectedFeatures);
    
    totalTPR = 0;
    
    %  5-fold cross-validation
    cv = cvpartition(Y, 'KFold', 5);
    for i = 1:cv.NumTestSets
        
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        % KNN classifier 
        classifier = fitcknn(X(trainIdx, selectedFeatures), Y(trainIdx), 'NumNeighbors', 5);
        
        % Predict 
        predictions = predict(classifier, X(testIdx, selectedFeatures));
        
        % Calculate TPR 
        for c = 1:numClasses
            actualPositives = sum(Y(testIdx) == c);
            truePositives = sum((Y(testIdx) == c) & (predictions == c));
            if actualPositives > 0
                totalTPR = totalTPR + truePositives / actualPositives;
            end
        end
    end
    
    
    ErrorRate = 1 - (totalTPR / (numClasses * cv.NumTestSets));
    
    fitness = alpha * ErrorRate + (1 - alpha) * (numSelected / numAll);
end
