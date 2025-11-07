% --- Step 1: Read color image ---
I = imread('4d.jpg');
R = im2double(I(:,:,1));
G = im2double(I(:,:,2));
B = im2double(I(:,:,3));
% --- Step 2: Otsu thresholding ---
tR = graythresh(R);
tG = graythresh(G);
tB = graythresh(B);
BW_R = imbinarize(R, tR);
BW_G = imbinarize(G, tG);
BW_B = imbinarize(B, tB);
% Try both AND and OR to see which gives better segmentation
BW_and = BW_R & BW_G & BW_B;
BW_or  = BW_R | BW_G | BW_B;
figure;
subplot(1,3,1), imshow(BW_R), title('Otsu - Red');
subplot(1,3,2), imshow(BW_G), title('Otsu - Green');
subplot(1,3,3), imshow(BW_and), title('Combined (AND)');
imwrite(BW_and, 'BW_mask_AND.png');
imwrite(BW_or, 'BW_mask_OR.png');
disp('Saved intermediate masks: BW_mask_AND.png and BW_mask_OR.png');
% --- Step 3: Load Ground Truth Image---
GT = imread('4dGT.png');
if size(GT,3) == 3
    GT = rgb2gray(GT);
end
GT = imbinarize(GT);
% Resize prediction to GT size
Pred = imresize(BW_and, size(GT));  % using AND mask
% --- Step 4: Fix inverted masks if needed ---
intersection = GT & Pred;
union = GT | Pred;
IoU = sum(intersection(:)) / sum(union(:));
if IoU < 0.01
    Pred = ~Pred;
    intersection = GT & Pred;
    union = GT | Pred;
    IoU = sum(intersection(:)) / sum(union(:));
end
fprintf('IoU = %.4f\n', IoU);
% --- Step 5: Visualization ---
figure;
imshowpair(GT, Pred);
title(sprintf('IoU using Otsu Thresholding = %.3f', IoU));
% --- Step 6: Save Predicted Mask and Comparison Output ---
% Save predicted binary mask
imwrite(Pred, 'Predicted_Mask.png');
disp('Saved predicted mask as Predicted_Mask.png');
% Save the visualization (IoU comparison)
saveas(gcf, 'IoU_Comparison.png');
disp('Saved IoU comparison image as IoU_Comparison.png');
