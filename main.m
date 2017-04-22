% Project Title: Object classification and Image segmentation Using Differential Evolution (DE) in MATLAB
%
%	Coded by: Mukesh Saraswat, saraswatmukesh@gmail.com 
%	Edited by: Himanshu Mittal, himanshu.mittal224@gmail.com

%% DE-based segmentation
img=imread('input.bmp');
imshow(img);
img_gray=rgb2gray(img);
%imshow(img_gray);
[T,variance1] = de(img_gray)
img_seg=im2bw(img_gray,T(1)/255);
figure, imshow(img_seg);

%% Feature Extraction

cc = bwconncomp(img_seg);
stats = regionprops(cc, 'Area','ConvexArea', 'Eccentricity', 'EulerNumber',...
'MajorAxisLength', 'MinorAxisLength','Perimeter','Solidity', 'Orientation');
F=zeros(cc.NumObjects,9);
for i=1:cc.NumObjects
    F(i,1)=stats(i,1).Area;
    F(i,2)=stats(i,1).ConvexArea;
    F(i,3)=stats(i,1).Eccentricity;
    F(i,4)=stats(i,1).EulerNumber;
    F(i,5)=stats(i,1).MajorAxisLength;
    F(i,6)=stats(i,1).MinorAxisLength;
    F(i,7)=stats(i,1).Perimeter;
    F(i,8)=stats(i,1).Solidity;
    F(i,9)=stats(i,1).Orientation;
end


%Recognition
%Phase 1: Training
load training_data
xdata = data(:,1:9);
group = data(:,10:10);
svmStruct = svmtrain(xdata,group);
%Phase 2: Testing
output = svmclassify(svmStruct,F);