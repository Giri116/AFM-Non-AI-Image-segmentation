%% --- Step 1: Read and preprocess image ---
I = imread('4d.jpg');   
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end
Igray = im2double(Igray);
%% --- Step 2: Load Ground Truth ---
GT = imread('4dGT.png');   
if size(GT,3) == 3
    GT = rgb2gray(GT);
end
GT = imbinarize(GT);
%% --- Step 3: Set optimization parameters ---
targetIoU = 0.85;              % desired IoU
bestIoU = 0;                  % initialize
bestMask = [];
bestParams = struct('seSize',0,'sigma',0);
seSizes = 1:2:10;             
sigmas = 0.5:0.5:3;
%% --- Step 4: Optimization loop ---
for seSize = seSizes
    for sigma = sigmas
        % --- Preprocessing ---
        I_filt = imgaussfilt(Igray, sigma);
        % --- Gradient magnitude ---
        [Gx, Gy] = imgradientxy(I_filt);
        gradmag = sqrt(Gx.^2 + Gy.^2);
        % --- Marker generation ---
        se = strel('disk', seSize);
        bw = imbinarize(I_filt, graythresh(I_filt));
        bw = imopen(bw, se);
        bw = imclose(bw, se);
        bw = imfill(bw, 'holes');
        sure_fg = bwareaopen(bw, 50);
        L = bwlabel(sure_fg);
        % --- Watershed ---
        gradmag2 = imimposemin(gradmag, L);
        Lw = watershed(gradmag2);
        segMask = Lw > 1;
        % --- Resize to GT and compute IoU ---
        segMask = imresize(segMask, size(GT));
        intersection = GT & segMask;
        union_area = GT | segMask;
        IoU = sum(intersection(:)) / sum(union_area(:));
        fprintf('se=%d, sigma=%.2f => IoU=%.4f\n', seSize, sigma, IoU);
        % --- Keep best ---
        if IoU > bestIoU
            bestIoU = IoU;
            bestMask = segMask;
            bestParams.seSize = seSize;
            bestParams.sigma = sigma;
        end
        % --- Stop early if target reached ---
        if IoU >= targetIoU
            fprintf('\nTarget IoU %.2f reached (%.4f) with se=%d, sigma=%.2f\n', ...
                targetIoU, IoU, seSize, sigma);
            break;
        end
    end
    if bestIoU >= targetIoU, break; end
end
%% --- Step 5: Display results ---
fprintf('\nBest IoU = %.4f (se=%d, sigma=%.2f)\n', bestIoU, ...
    bestParams.seSize, bestParams.sigma);
figure;
subplot(1,3,1), imshow(I), title('Original');
subplot(1,3,2), imshow(bestMask), title('Best Watershed Mask');
subplot(1,3,3), imshowpair(GT, bestMask);
title(sprintf('IoU using Watershed = %.3f ', bestIoU));
%% --- Step 6: Save output mask ---
imwrite(bestMask, 'Best_Watershed_Mask.png');
disp(' Best segmented mask saved as Best_Watershed_Mask.png');
