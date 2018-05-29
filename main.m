%% Initialize
clear;clc; close all;
% 1 = meningioma, 2 = glioma, 3 = pituitary tumor

load('dataset128.mat');
 
trainDataSize=height(trainData);
testDataSize=height(testData);

doTrainingAndEval = false;

% for i=1:trainDataSize
%     trainData4d(:,:,1,i)= trainData.imageSource{i,1};  
% end
% 
% for i=1:testDataSize
%     testData4d(:,:,1,i)= testData.imageSource{i,1};
% end
% 
% trainLabel= categorical( trainData.label); 
%testLabel= categorical(testData.label);

%% Read Image For Test

idx=randperm(trainDataSize,6);
for i = 1:numel(idx)
    I=trainData4d(:,:,1,idx(i));
    I = insertShape(I, 'Rectangle', trainData.borderRectangle{idx(i)}, 'LineWidth',1);
    
    tumorType={ 'Meningioma', 'Glioma', 'Pituitary' };
    tumorName= tumorType(trainData.label(idx(i)));
    
    subplot(2,3,i);
    imshow(I);
    title("img" + num2str(idx(i)) + "=" + tumorName);
end

%% Init Convolutional Neural Network

layers = [ ...
    imageInputLayer([128 128 1], 'Name','input')
    convolution2dLayer(5,5, 'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,1,'Padding','same','Stride',2,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    convolution2dLayer(3,1,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool')
    %averagePooling2dLayer(2,'Stride', 2, 'Name', 'avpool')
    
    fullyConnectedLayer(3, 'Name', 'FC')
    reluLayer('Name', 'relu_out')
    
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

options = trainingOptions('sgdm',...
    'MaxEpochs',6,...%    'ExecutionEnvironment','gpu',...
    'Shuffle','every-epoch',...
    'ValidationData',{testData4d,testLabel},...
    'ValidationFrequency',20,...
    'VerboseFrequency',20);

%% Display Graph

lgraph = layerGraph(layers);
figure
plot(lgraph)

%% Train Network
if doTrainingAndEval
     net = trainNetwork(trainData4d,trainLabel,lgraph,options);
end

net

%% Display Convolution Layer
act1 = activations(net,trainData4d(:,:,1,idx(1)),'conv_1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
figure;
montage(mat2gray(act1),'Size',[2 3]);
title('Convolution Layer 1');
%%
% Classify the validation images and calculate the accuracy.
predictedLabels = classify(net,testData4d);
accuracy = mean(predictedLabels == testLabel)