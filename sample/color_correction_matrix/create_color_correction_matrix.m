%{
This script is to create a color correction matrix (CCM). 
As a reference, see:
 https://www.mathworks.com/help/images/correct-colors-using-color-correction-matrix.html
%}

close all;
clear all;


%% USER SETTING
addpath(strcat(pwd, '/sample/color_correction_matrix'));
save_ccm = true;
input_file_path = 'data/sample_image/processed_rgb.png';

%% Detect Macbeth chart and evaluate colors.

p = genpath(pwd);  % pwd should return the root of this repo.
addpath(p);
img = imread(input_file_path);

imshow(img);

blackPoint = drawpoint;
whitePoint = drawpoint;
darkSkinPoint = drawpoint;
bluishGreenPoint = drawpoint;

cornerPoints = [blackPoint.Position;
    whitePoint.Position;
    darkSkinPoint.Position;
    bluishGreenPoint.Position];

chart = colorChecker(img, "RegistrationPoints", cornerPoints);

%% Assess results and construct CCM
[colorTable, ccm] = measureColor(chart);
disp(colorTable)
disp(ccm)
displayChart(chart);

new_img = imapplymatrix(ccm(1:3,:)',img,ccm(4,:));
figure('Name', 'Corrected Image');
imshow(new_img)

if save_ccm
    save_file_path = strcat(pwd, '/sample/color_correction_matrix/ccm.mat');
    save(save_file_path, 'ccm');
    disp('ccm.mat is saved')
end