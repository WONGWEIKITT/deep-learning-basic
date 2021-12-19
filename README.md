clc
clear all
close all

digitDatasetPath = fullfile('C:\Users\ASUS\Desktop\resize image(2)');%%set the datapath acordingly
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');


% [imdsTrain,imdsValidation] = splitEachLabel(imds,0.8,0.2,'randomized');
[imdsTrain,imdsValidation, imdstest] = splitEachLabel(imds,0.8,0.2);

 
labelCount = countEachLabel(imds);

net = alexnet;


 layers = [imageInputLayer([128 128 1])
   net(2:end-3)
   fullyConnectedLayer(2)
   softmaxLayer
   classificationLayer()
   ];
%%%%%%%%%%%%%%%%%%%
% 
% inputSize = net.Layers(1).InputSize;
% 
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
% 
% augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% 


%%%%%%%%%%%%%%%%%%%%%%

% training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',50, ...
    'Plots','training-progress');
   % 'Verbose',false, ...
   
   
    train = trainNetwork(imdsTrain,layers,options);
