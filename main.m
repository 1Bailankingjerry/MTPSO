clc;clear;
tic;%计时
%Get data
load("E:\科研\dataset\lung.mat");
X=zscore(X);
% 分割数据集为训练集和测试集
cv = cvpartition(Y, 'HoldOut', 0.1);
X_train = X(cv.training, :);
Y_train = Y(cv.training);
X_test = X(cv.test, :);
Y_test = Y(cv.test);

%%%%%%%%%%%%%%%%%%%%%%%%%Task Generation Strategy%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate the relieff
[ranks, weights]=relieff(X_train, Y_train, 5);      %relieff函数中ranks显示根据排名列出的预测变量编号，weights以与预测变量相同的顺序给出权重值
% find the knee point
[sorted_weights, ~] = sort(weights, 'descend');
% 计算连线的斜率和截距
x = 1:length(weights);  % 横坐标只是标号
y = sorted_weights;     % 降序排列的权重作为纵坐标
slope = (y(end) - y(1)) / (x(end) - x(1));
intercept = y(1) - slope * x(1);
% 计算每个点到连线的距离
distances = abs(-slope * x + y - intercept) / sqrt(slope^2 + 1);
% 找到距离最远的点，该点的权重即为划分阈值
[~, maxDistIdx] = max(distances);
threshold_weight = y(maxDistIdx);
% 使用原始权重（未排序）与阈值比较来划分Promising和Remaining Sets
promising_idx = find(weights >= threshold_weight);
remaining_idx = find(weights < threshold_weight);
% 在选择特征之前，移除权重为负的特征索引
promising_idx = promising_idx(weights(promising_idx) > 0);
remaining_idx = remaining_idx(weights(remaining_idx) > 0);
% 图像表示
% figure;
% plot(x, y, 'b-o'); % 绘制排序后的权重
% hold on;
% plot(x, slope * x + intercept, 'r--'); % 绘制最高点到最低点的连线
% plot(x(maxDistIdx), threshold_weight, 'g*'); % 标记距离最远的点
% xlabel('Feature Number (sorted by weight)');
% ylabel('Weight');
% title('Feature Weights and Max Distance Point');
% legend('Feature Weights', 'Line from Max to Min weight', 'Max Distance Point');
% hold off;

% form related FS tasks
% 计算两个集合的平均权重
avg_weight_promising = mean(abs(weights(promising_idx)));
avg_weight_remaining = mean(abs(weights(remaining_idx)));
% 计算选择概率
p_promising = avg_weight_promising / (avg_weight_promising + avg_weight_remaining);
p_remaining = 1 - p_promising;
T = 5; % 生成T个相关任务
num_features = length(weights); % 特征总数
tasks = cell(1, T); % 初始化任务存储变量
for  t = 1:T
    % 对于每个任务，根据概率选择特征
    selected_promising = randsample(promising_idx, round(p_promising * num_features), true);
    selected_remaining = randsample(remaining_idx, round(p_remaining * num_features), true);
    % 合并选择的特征并存储为一个任务
    tasks{t} = union(selected_promising, selected_remaining);
end

%%%%%%%%%%%%%%%%%%%%%%%%%Multitasking Optimization Method%%%%%%%%%%%%%%%%%%
alpha = 0.9; % 适应度函数中的权重
rmp = 0.6; % 随机配对概率
G = 6; % 未变化代数阈值
numIterations = 10; % 迭代次数
cr = rand(); % 交叉率，用于知识迁移
numFeatures = size(X_train, 2); % 特征数量
numParticles = min(floor(numFeatures / 3), 200); % 粒子数量
c1 = 1.49445; % 认知参数
c2 = 1.49445; % 社会参数
cnt=0;
knowcnt=0;
%%%%%%
particles = cell(T, 1);
globalBests = cell(T, 1);
% Initialize particles for each task
for t = 1:T
    featureSubset = tasks{t}; % 任务t
    numFeaturesInTask = length(featureSubset);
    % Initialize particles for task t
    for i = 1:numParticles
        % 初始化二进制
        particlePosition = false(1, numFeatures);
        particlePosition(featureSubset) = rand(1, numFeaturesInTask) > 0.6;

        particles{t}(i).position = particlePosition;
        particles{t}(i).velocity = zeros(1, numFeatures);
        particles{t}(i).fitness = Inf;
        particles{t}(i).pbest = particlePosition;
        particles{t}(i).pbestFitness = Inf;
    end
    % Initialize the global best for task t
    globalBests{t} = struct('position',particles{t}(1).position, 'fitness', Inf);  % Initialize each cell as a struct
end
%%%%%%
% Optimization Loop
for iter = 1:numIterations
    fprintf('第 %d 次迭代\n', iter);
    w=0.9-0.5*(iter/numIterations);
    for t = 1:T
        cnt=0;  %每个任务更新cnt
        fprintf('第 %d 次迭代 任务%d\n', iter,t);
        % 从tasks中获取当前任务的特征子集
        featureSubset = tasks{t};
        for i = 1:numParticles
            %保存该粒子位置
            currentPosition=particles{t}(i).position;
            % 更新粒子速度
            r1 = rand();
            r2 = rand();
            particles{t}(i).velocity = w * particles{t}(i).velocity + ...
                c1 * r1 .* (particles{t}(i).pbest - particles{t}(i).position) + ...
                c2 * r2 .* (globalBests{t}.position - particles{t}(i).position);
            % 预更新粒子位置
            newPosition = currentPosition + particles{t}(i).velocity;

            %             particles{t}(i).position = particles{t}(i).position + particles{t}(i).velocity;

            % 适用于特征选择的二值化处理，确保只选择当前任务相关的特征
            selectedFeatures = false(1, numFeatures); % Initialize with false
            selectedFeatures(featureSubset) = particles{t}(i).position(featureSubset) > 0.6;
            % 检查是否有特征被选中
            if sum(selectedFeatures) == 0
                % 如果没有特征被选中，保持粒子在原位置，不更新粒子状态，直接跳过该粒子的后续处理
                fprintf('任务 %d, 粒子 %d: 没有特征被选中，保持当前状态不变。\n', t, i);
                continue;  % 跳过该粒子的后续处理
            end
            % 有特征被选中，更新粒子位置
            particles{t}(i).position = newPosition;
            % 计算适应度
            %           selectedFeatures = particles{t}(i).position(featureSubset);
            particles{t}(i).fitness = calculateFitness(selectedFeatures, X_train, Y_train, alpha);

            % 更新个体最佳
            if isempty(particles{t}(i).pbest) || particles{t}(i).fitness < particles{t}(i).pbestFitness
                particles{t}(i).pbest = particles{t}(i).position;
                particles{t}(i).pbestFitness = particles{t}(i).fitness;
            end

            % 更新全局最优解
            if isempty(globalBests{t}) || particles{t}(i).fitness < globalBests{t}.fitness
                globalBests{t}.position = particles{t}(i).position;
                globalBests{t}.fitness = particles{t}(i).fitness;
                cnt=0;
            else
                cnt=cnt+1;
            end
        end

        % 知识迁移部分
        if rand() <= rmp
            %  tournament selection mechanism
            otherTasks = setdiff(1:T, t);
            selectedTaskIdx = otherTasks(randi(length(otherTasks)));
            gbest_m = globalBests{selectedTaskIdx}.position; % Position of the selected gbest


            cr = rand(1, numFeatures);

            % 生成新gbest
            gbest_new = cr .* globalBests{t}.position + (1 - cr) .* gbest_m;
            % G代未改变更新gbest平均值
            if cnt>6
                gbest_sum = zeros(1, numFeatures);
                for i = 1:T
                    gbest_sum = gbest_sum + globalBests{i}.position;
                end
                gbest_new = gbest_sum / T;
                cnt=0;%更新计数器
                knowcnt=knowcnt+1;
            end
            % 检查是否有特征被选中
            selectedFeatures = gbest_new > 0.6;
            if sum(selectedFeatures) == 0
                % 如果没有特征被选中，则不更新全局最优
                fprintf('没有特征被选中，跳过更新全局最优解。\n');
                continue;
            else
                %
                newFitness = calculateFitness(gbest_new > 0.6, X_test, Y_test, alpha);
                %             if newFitness < globalBests{t}.fitness
                %更新gbest
                globalBests{t}.position = gbest_new > 0.6;
                globalBests{t}.fitness = newFitness;
                %             end
            end
        end % 知识迁移部分end
    end
end

% 输出每个任务的最佳特征子集
BestFS = cell(1, T);
for t = 1:T
    BestFS{t} = globalBests{t};
end

toc
%
maxAccuracy = 0; % 初始化最大正确率为0
maxAccuracyFeaturesCount = 0; % 初始化最大正确率对应的特征个数为0
for t = 1:T
    % 提取对应任务的最佳特征子集的索引
    selectedFeatures = BestFS{t}.position>0.6;
    % 计算选取的特征个数
    numSelectedFeatures = sum(selectedFeatures);
    % 该特征子集的分类正确率
    accuracy = evaluateFeatureSubset(selectedFeatures, X_test, Y_test, 1);
    fprintf('任务 %d 的特征子集中选取的特征个数: %d\n', t, numSelectedFeatures);
    fprintf('任务 %d 的特征子集分类正确率: %.2f%%\n', t, accuracy * 100);
    if accuracy > maxAccuracy
        maxAccuracy = accuracy;
        maxAccuracyFeaturesCount = numSelectedFeatures;
    end
end
disp(['MTPSO运行时间:',num2str(toc)]);
fprintf('ACC: %.2f%%\n',maxAccuracy*100);
fprintf('特征数量: %d\n', maxAccuracyFeaturesCount);
