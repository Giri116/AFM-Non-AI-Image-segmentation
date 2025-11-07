%% --- Step 1: Read and preprocess image ---
I = imread('4d.jpg');       
figure, imshow(I), title('Original Image');
if size(I,3) ~= 3
    I = cat(3, I, I, I);    
end 
%% --- Step 2: Apply SLIC Superpixel Segmentation ---
numSuperpixels = 300;       % adjust number of superpixels
compactness = 20;           % balance color vs. spatial proximity
[L, N] = superpixels(I, numSuperpixels, 'Compactness', compactness);
fprintf('Generated %d superpixels.\n', N);
%% --- Step 3: Create region boundaries overlay ---
BW = boundarymask(L);
figure;
imshow(imoverlay(I, BW, 'cyan'));
title('SLIC Superpixel Boundaries');
%% --- Step 4: Region merging based on color similarity ---
% Compute mean color for each superpixel
meanColors = zeros(N,3);
for k = 1:N
    mask = (L == k);
    for c = 1:3
        channel = I(:,:,c);
        meanColors(k,c) = mean(channel(mask));
    end
end
% Simple merging heuristic: cluster similar colors (K-means)
Kmerge = 3;  % number of merged regions
[idx_merge, ~] = kmeans(meanColors, Kmerge, 'Replicates', 3);
% Create merged segmentation map
mergedSeg = zeros(size(L));
for k = 1:N
    mergedSeg(L == k) = idx_merge(k);
end
figure;
imshow(label2rgb(mergedSeg)), title('Merged Superpixel Regions (SLIC + K-means)');
%% --- Step 5: Choose one or more regions as object (e.g., region 1) ---
Pred = mergedSeg == 1;
% Smooth mask slightly
Pred = imclose(Pred, strel('disk', 3));
Pred = imfill(Pred, 'holes');
figure;
imshow(Pred);
title('Predicted Foreground Mask (Merged Region 1)');
%% --- Step 6: Load Ground Truth and compute IoU ---
GT = imread('4dGT.png');  
if size(GT,3) == 3
    GT = rgb2gray(GT);
end
GT = imbinarize(GT);
GT = imresize(GT, size(Pred));
% Compute IoU
intersection = GT & Pred;
union_area = GT | Pred;
IoU = sum(intersection(:)) / sum(union_area(:));
fprintf('IoU = %.4f\n', IoU);
%% --- Step 7: Visualize comparison ---
figure;
subplot(1,3,1), imshow(GT), title('Ground Truth');
subplot(1,3,2), imshow(Pred), title('Predicted Mask');
subplot(1,3,3), imshowpair(GT, Pred);
title(sprintf('IoU using SLIC Superpixel = %.3f', IoU));
%% --- Step 8: Save output mask ---
imwrite(Pred, 'SLIC_PredictedMask.png');
disp('Saved as SLIC_PredictedMask.png');
