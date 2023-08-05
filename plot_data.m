% plot 

data = readmatrix('dataset.txt');

% Extract the features and class labels
X = data(:, 1:2);
y = data(:, 3:5);

% Plot the data
scatter(X(:, 1), X(:, 2), 25, y, 'filled');

% Set the title and axis labels
title('Linearly Separable Dataset with 3 Classes');
xlabel('Feature 1');
ylabel('Feature 2');


% Extract the features and class labels
X = data(:, 1:2);
y = data(:, 3:5);

% Find the indices of the samples for each class
idx_class1 = find(y(:, 1));
idx_class2 = find(y(:, 2));
idx_class3 = find(y(:, 3));

% Plot the data points
scatter(X(idx_class1, 1), X(idx_class1, 2), 'r', 'filled');
hold on;
scatter(X(idx_class2, 1), X(idx_class2, 2), 'g', 'filled');
scatter(X(idx_class3, 1), X(idx_class3, 2), 'b', 'filled');


% Set the title and axis labels
title('Linearly Separable Dataset with 3 Classes');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');


% Extract the features and class labels
X = data(:, 1:2);
y = data(:, 3:5);

% Plot the data points for each class separately
figure;
subplot(1, 3, 1);
scatter(X(y(:, 1) == 1, 1), X(y(:, 1) == 1, 2), 'r', 'filled');
hold on;
scatter(X(y(:, 1) == 0, 1), X(y(:, 1) == 0, 2), 'k', 'filled');
title('Class 1 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(1, 3, 2);
scatter(X(y(:, 2) == 1, 1), X(y(:, 2) == 1, 2), 'g', 'filled');
hold on;
scatter(X(y(:, 2) == 0, 1), X(y(:, 2) == 0, 2), 'k', 'filled');
title('Class 2 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(1, 3, 3);
scatter(X(y(:, 3) == 1, 1), X(y(:, 3) == 1, 2), 'b', 'filled');
hold on;
scatter(X(y(:, 3) == 0, 1), X(y(:, 3) == 0, 2), 'k', 'filled');
title('Class 3 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');


% Extract the features and class labels
X = data(:, 1:2);
y = data(:, 3:5);

% Train a linear SVM for each class vs the rest
models = cell(3, 1);
for i = 1:3
    y_binary = double(y(:, i) == 1);
    models{i} = fitcsvm(X, y_binary, 'KernelFunction', 'linear');
end

% Plot the data points for each class separately
figure;
subplot(1, 3, 1);
scatter(X(y(:, 1) == 1, 1), X(y(:, 1) == 1, 2), 'r', 'filled');
hold on;
scatter(X(y(:, 1) == 0, 1), X(y(:, 1) == 0, 2), 'k', 'filled');
title('Class 1 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(1, 3, 2);
scatter(X(y(:, 2) == 1, 1), X(y(:, 2) == 1, 2), 'g', 'filled');
hold on;
scatter(X(y(:, 2) == 0, 1), X(y(:, 2) == 0, 2), 'k', 'filled');
title('Class 2 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');

subplot(1, 3, 3);
scatter(X(y(:, 3) == 1, 1), X(y(:, 3) == 1, 2), 'b', 'filled');
hold on;
scatter(X(y(:, 3) == 0, 1), X(y(:, 3) == 0, 2), 'k', 'filled');
title('Class 3 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');

% Plot the hyperplanes for each class vs the rest
x1 = linspace(1, 6);
for i = 1:3
    w = models{i}.Beta;
    b = models{i}.Bias;
    fprintf('Class %d vs Rest: %.2f*x1 + %.2f*x2 + (%.2f) = 0\n', i, w(1), w(2), b);
    x2 = -(w(1)*x1 + b)/w(2);
    subplot(1, 3, i);
    plot(x1, x2, 'k--', 'LineWidth', 2);
end

% Print the support vectors for each class
for i = 1:3
    fprintf('Support vectors for Class %d:\n', i);
    disp(models{i}.SupportVectors);
end

sgtitle('One Against All Multi-Class SVM Decision Boundaries');


% Define the line equation
x1 = linspace(1, 6);
x2 = (1.06*x1 - 3.23)/0.39;

% Plot the line on the Class 2 vs Rest subplot
figure;
subplot(1, 3, 2);
scatter(X(y(:, 2) == 1, 1), X(y(:, 2) == 1, 2), 'g', 'filled');
hold on;
scatter(X(y(:, 2) == 0, 1), X(y(:, 2) == 0, 2), 'k', 'filled');
plot(x1, x2, 'm-', 'LineWidth', 2);
title('Class 2 vs Rest');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 2', 'Rest', 'Decision boundary');
