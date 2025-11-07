% ---- Step 1: Read Image ----
I = imread('4d.jpg');
if size(I,3) == 3
    I = rgb2gray(I);
end
I = im2double(I);
% ---- Step 2: Parameters ----
windowSize = 95;   
C = 0.02; 
% ---- Step 3: Local Mean Thresholding ----
localMean = imfilter(I, fspecial('average', windowSize), 'replicate');
BW_mean = I > (localMean - C);
% ---- Step 4: Local Gaussian Thresholding ----
sigma = 0.5 * windowSize;   % standard deviation for Gaussian
gaussianFilter = fspecial('gaussian', windowSize, sigma);
localGauss = imfilter(I, gaussianFilter, 'replicate');
BW_gauss = I > (localGauss - C);
% ---- Step 5: Display Results ----
figure;
subplot(2,3,1), imshow(I), title('Original Grayscale Image');
subplot(2,3,2), imshow(localMean, []), title('Local Mean');
subplot(2,3,3), imshow(BW_mean), title('Adaptive (Local Mean)');
subplot(2,3,4), imshow(localGauss, []), title('Local Gaussian Weighted');
subplot(2,3,5), imshow(BW_gauss), title('Adaptive (Gaussian)');
% ---- Step 6: IoU Calculation ----
GT = imread('4dGT.png');  % Ground Truth mask
if size(GT,3) == 3
    GT = rgb2gray(GT);
end
GT = imbinarize(GT);
% Resize predictions to match GT
BW_mean_resized = imresize(BW_mean, size(GT));
BW_gauss_resized = imresize(BW_gauss, size(GT));
% --- IoU for Local Mean ---
intersection_mean = GT & BW_mean_resized;
union_mean = GT | BW_mean_resized;
IoU_mean = sum(intersection_mean(:)) / sum(union_mean(:));
% --- IoU for Gaussian ---
intersection_gauss = GT & BW_gauss_resized;
union_gauss = GT | BW_gauss_resized;
IoU_gauss = sum(intersection_gauss(:)) / sum(union_gauss(:));
% ---- Step 7: Display IoU results ----
fprintf('IoU (Local Mean)     = %.4f\n', IoU_mean);
fprintf('IoU (Local Gaussian) = %.4f\n', IoU_gauss);
% ---- Step 8: Visualization of overlap ----
figure;
subplot(1,2,1);
imshowpair(GT, BW_mean_resized);
title(sprintf('IOU using Adaptive Thresholding — IoU = %.3f', IoU_mean));
subplot(1,2,2);
imshowpair(GT, BW_gauss_resized);
title(sprintf('Overlap (Gaussian) — IoU = %.3f', IoU_gauss));
% ---- Step 9: Save Outputs ----
imwrite(BW_mean, 'Adaptive_Mean_Mask.png');
imwrite(BW_gauss, 'Adaptive_Gaussian_Mask.png');
saveas(gcf, 'IoU_Comparison.png');
disp(' Results saved:');
disp('   - Adaptive_Mean_Mask.png');
disp('   - Adaptive_Gaussian_Mask.png');
disp('   - IoU_Comparison.png');
