% file_path = "/Users/kotarosonoda/Documents/pythonProject/archive/AlphaISP - AWB Dataset/AlphaISP - AWB Dataset/PNG Data/AlphaISP_2592x1536_8bits_Scene5.png";

file_path = "/Users/kotarosonoda/Documents/pythonProject/processed_rgb.png";

img = imread(file_path);

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
[colorTable, ccm] = measureColor(chart);
disp(ccm)
displayChart(chart);


new_img = imapplymatrix(ccm(1:3,:)',img,ccm(4,:));
imshow(new_img)