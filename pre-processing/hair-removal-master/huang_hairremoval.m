
% % PH2 dataset
% PH2datasetPath = 'E:\medi_vision\skin lesion\PH2\PH2Dataset\';
% imgdirNameList = dir(fullfile(PH2datasetPath, 'PH2 Dataset Images'));
% folder_nums = size(imgdirNameList, 1);
% for i=3:folder_nums
%     imgdirName = imgdirNameList(i,1).name;
%     imgPath = fullfile(PH2datasetPath, "edge_removed", strcat(imgdirName, '.bmp'));
%     outputPath = fullfile(PH2datasetPath, 'huang_hairremoval', strcat(imgdirName, '.bmp'))
%     I = imread(imgPath);
%     [M]=ncuLineCloseMatch(im2gray(I),12);
%     [K]=stdDilateDarkest(255*M,255*0.75,255*0.65,40);
%     [K]=stdDilateColorDist3(I,K,(K(:,:,1)<255),40,0.5,25,1);
%     K=K(:,:,1)>=255;
%     [J]=HairRemovMed(I,K,15);
%     imwrite(J, outputPath);
% end

% ISIC2016 dataset
parpool(7);
ISIC2016datasetPath = 'E:\medi_vision\skin lesion\ISIC2016_test\';
imgNameList = dir(fullfile(ISIC2016datasetPath, 'resized_hairremoval'));
img_nums = size(imgNameList, 1);
parfor i=3:img_nums
    imgName = imgNameList(i,1).name;
    imgPath = fullfile(ISIC2016datasetPath, "resized_hairremoval", imgName);
    outputPath = fullfile(ISIC2016datasetPath, 'huang_hairremoval', imgName)
    I = imread(imgPath);
    [M]=ncuLineCloseMatch(im2gray(I),12);
    [K]=stdDilateDarkest(255*M,255*0.75,255*0.65,40);
    [K]=stdDilateColorDist3(I,K,(K(:,:,1)<255),40,0.5,25,1);
    K=K(:,:,1)>=255;
    [J]=HairRemovMed(I,K,15);
    imwrite(J, outputPath);
end

% % ISIC2017 dataset
% ISIC2017datasetPath = 'E:\medi_vision\skin lesion\ISIC2017\';
% imgNameList = dir(fullfile(ISIC2017datasetPath, 'test_resized'));
% img_nums = size(imgNameList, 1);
% for i=3:img_nums
%     imgName = imgNameList(i,1).name;
%     imgPath = fullfile(ISIC2017datasetPath, "test_resized", imgName);
%     outputPath = fullfile(ISIC2017datasetPath, 'huang_hairremoval', imgName)
%     I = imread(imgPath);
%     [M]=ncuLineCloseMatch(im2gray(I),12);
%     [K]=stdDilateDarkest(255*M,255*0.75,255*0.65,40);
%     [K]=stdDilateColorDist3(I,K,(K(:,:,1)<255),40,0.5,25,1);
%     K=K(:,:,1)>=255;
%     [J]=HairRemovMed(I,K,15);
%     imwrite(J, outputPath);
% end

% pnamei='E:\medi_vision\skin lesion\preprocessing\hair-removal-master\';
% pnamej=pnamei;
% pnamek=[];
% fname='2006-124-2.tif';
% 
% 
% 
% 
% 
% I=imread([pnamei,fname]);
% 
% 
% %[M]=ncuLineCloseMatch(rgb2gray(I),12);
% [M]=ncuLineCloseMatch(im2gray(I),12);
% [K]=stdDilateDarkest(255*M,255*0.75,255*0.65,40);
% [K]=stdDilateColorDist3(I,K,(K(:,:,1)<255),40,0.5,25,1);
% K=K(:,:,1)>=255;
% [J]=HairRemovMed(I,K,15);
% 
% fname='2006-124-4.tif';
% imwrite(J,[pnamej,fname]);