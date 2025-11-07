%% ---- Step 1: Read input image ----
I = imread('4d.jpg');
I = im2double(I);
imshow(I); title('Original Image');
%% ---- Step 2: Reshape for K-means ----
data = reshape(I, [], 3);   % N x 3 matrix of RGB values
%% ---- Step 3: K-means clustering ----
K = 3;
[cluster_idx, ~] = kmeans(data, K, 'Distance', 'sqEuclidean', 'Replicates', 3);
pixel_labels = reshape(cluster_idx, size(I,1), size(I,2));
%% ---- Step 4: Visualize segmentation ----
figure;
imshow(label2rgb(pixel_labels));
title('K-Means Segmentation (K=3)');
%% ---- Step 5: Load Ground Truth Image ----
GT = imread('4dGT.png');
if size(GT,3) == 3
    GT = rgb2gray(GT);
end
GT = imbinarize(GT);
GT = imresize(GT, size(pixel_labels));
%% ---- Step 6: Test IoU for each cluster ----
bestIoU = 0;
bestMask = [];
for k = 1:K
    Pred = (pixel_labels == k);
    % Optional smoothing to clean noise (comment out if not needed)
    Pred = imopen(Pred, strel('disk', 2));
    % Compute IoU
    intersection = GT & Pred;
    union = GT | Pred;
    IoU = sum(intersection(:)) / sum(union(:));
    fprintf('Cluster %d -> IoU = %.4f\n', k, IoU);
    % Keep best
    if IoU > bestIoU
        bestIoU = IoU;
        bestMask = Pred;
        bestCluster = k;
    end
end
%% ---- Step 7: Combine multiple clusters if needed (optional boost) ----
% Try combining clusters if best IoU < 0.8
if bestIoU < 0.8
    fprintf('\nTrying combinations of clusters to boost IoU...\n');
    for c1 = 1:K-1
        for c2 = c1+1:K
            Pred = (pixel_labels == c1) | (pixel_labels == c2);
            intersection = GT & Pred;
            union = GT | Pred;
            IoU = sum(intersection(:)) / sum(union(:));
            fprintf('Clusters [%d %d] -> IoU = %.4f\n', c1, c2, IoU);

            if IoU > bestIoU
                bestIoU = IoU;
                bestMask = Pred;
                bestCluster = [c1 c2];
            end
        end
    end
end
%% ---- Step 8: Display best result ----
fprintf('\n Best IoU = %.4f using cluster(s): %s\n', bestIoU, mat2str(bestCluster));

figure;
subplot(1,3,1), imshow(GT), title('Ground Truth');
subplot(1,3,2), imshow(bestMask), title('Best Predicted Mask');
subplot(1,3,3), imshowpair(GT, bestMask);
title(sprintf('IoU using K-means clustering = %.3f', bestIoU));
%% ---- Step 9: Save predicted binary mask ----
imwrite(bestMask, 'BestMask_KMeans.png');
disp('Saved best mask as BestMask_KMeans.png');
