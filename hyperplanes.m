% Load the dataset
data = load('dataset.txt');

% Extract features and labels
X = data(:, 1:2);
Y = data(:, 3:5);

% One-vs-rest approach
figure
for i = 1:size(Y, 2)
    subplot(1, size(Y, 2), i)
    % Select positive and negative classes
    y = Y(:, i);
    neg_idx = find(y == 0);
    pos_idx = find(y == 1);
    
    % Train SVM
    svm = fitcsvm(X, y, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'ClassNames', [0, 1]);
    
    % Find support vectors
    sv = svm.SupportVectors;
    sv_labels = svm.SupportVectorLabels;
    sv_pos_idx = find(sv_labels == 1);
    sv_neg_idx = find(sv_labels == -1);
    sv_pos = sv(sv_pos_idx, :);
    sv_neg = sv(sv_neg_idx, :);
    
    % Plot data points and decision boundary
    gscatter(X(:, 1), X(:, 2), y);
    hold on
    plot(sv_pos(:, 1), sv_pos(:, 2), 'go', 'MarkerSize', 10, 'LineWidth', 2)
    plot(sv_neg(:, 1), sv_neg(:, 2), 'ro', 'MarkerSize', 10, 'LineWidth', 2)
    plotBoundary(svm, [min(X(:, 1)) max(X(:, 1))], [min(X(:, 2)) max(X(:, 2))], 'LineWidth', 2);
    title(sprintf('Class %d vs Rest', i))
    legend('Negative', 'Positive', 'Support vectors', 'Hyperplane')
end
