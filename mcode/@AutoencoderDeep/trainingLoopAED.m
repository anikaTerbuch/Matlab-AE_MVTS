function trainingLoopAED(obj,data,gradientFunction)
% train, AED, varying length, ADAM
%
% Purpose : This fuction trains an AutoencoderDeep on the data passed to
% the function with the cost function specified in the gradient
% function.
% Code optimized for training on multiple GPUs
%
% Syntax : trainingLoopAED(obj,data,gradientFunction)
%
% Input Parameters :
% - obj: AutoencoderDeep
% - data: data the obj should be trained on
% - gradientFunction: function handle of function which should be used to
% calculate the gradients during training
%
% Return Parameters :
%
% Description :
% This function converts the cell arrays of data into dl-arrays (Matlab's
% data type for training custom neural networks) or gpu
% arrays (when training is done on a gpu). In one dl-array the data needs
% to have constant length. Because of this, the samples used for training
% are sampled and then devided into the samples which form mini-batches.
% Then the samples are resampled to the shortest data-sequence of the
% mini-batch. After that the dl-arrays are used during training. The
% training is perfomred with the learning-parameters defined in the
% hyperparamter struct and the gradient function (function for gradient
% calculation of the loss function) specified by the autoenocder type. As
% learning algorithm ADAM is used. After the training of the object is
% completed the property Trained is set to true.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{20-Jan-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%
%% Hyperparameters

% extract some hyperparameters for better readability of the code
numEpochs=obj.Hyperparameters.Hyperparameters.NumberEpoch;
miniBatchSize=obj.Hyperparameters.Hyperparameters.MiniBatchSize;
executionEnvironment=obj.Hyperparameters.Hyperparameters.ExecutionEnvironment;
learningRate=obj.Hyperparameters.Hyperparameters.LearningRate;


%% Preperation for training

% initializations
avgGradientsEncoder=[];
avgGradientsSquaredEncoder=[];
avgGradientsDecoder=[];
avgGradientsSquaredDecoder=[];

%% Training the network (encoder and decoder sumutaneously)
% using custom training loops

% For each iteration in an epoch
% - get the net mini-batch from the training set
% - convert the mini-batch to a dlarray with the dimensions (CBT: channel,
%    batch observations, time or sequence)
% - for GPU training: convert the dlarray to a GPU array
% - evaluate the model gradients using the dlfeval and modelGradient
% functions
% update the networks learnables and the average gradients of the network
% using the adamupdate function

% get the data in dl-array by mini-batches
[cellOfDlArrays, numIterations]=...
    AutoencoderDeep.varSeqLen2dlarray(data, miniBatchSize,executionEnvironment);

% set up a pool for parallel computing
if canUseGPU
    executionEnvironment = "gpu";
    numberOfGPUs = gpuDeviceCount("available");
    pool = parpool(numberOfGPUs);
else
    executionEnvironment = "cpu";
end


% loop over the epochs
for epoch=1:numEpochs
    % loop over the iterations

    for iteration = 1:numIterations

        %% Calculating the gradients with automatic differentiation
        % deep learning function evaluation for automatic differentiation
        % passed arguments:
        %  - handle to function which calculates the gradients
        %  - arguments of the passed function with the function handle
        % (modelGradient(encoderNet, decoderNet, XBatch)

        % cost function of VAE has two terms which are weighted with a
        % weighting parameter

        if obj.Hyperparameters.Hyperparameters.AutoencoderType=="AE"

            [encoderGrad, decoderGrad] = dlfeval(...
                (gradientFunction), obj.EncoderNet, obj.DecoderNet, ...
                cellOfDlArrays{iteration});

        elseif obj.Hyperparameters.Hyperparameters.AutoencoderType=="VAE"
            [encoderGrad, decoderGrad] = dlfeval(...
                (gradientFunction), obj.EncoderNet, obj.DecoderNet, ...
                cellOfDlArrays{iteration}, ...
                obj.Hyperparameters.Hyperparameters.WeightingKL);
        end

        %% Updating the learnables using adaptive moment estimation (ADAM)
        % adam update on the learnables of the decoder:
        [obj.DecoderNet.Learnables, avgGradientsDecoder, ...
            avgGradientsSquaredDecoder] = ...
            adamupdate(obj.DecoderNet.Learnables, decoderGrad, ...
            avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, ...
            learningRate);

        % adam update on the learnables of the encoder
        [obj.EncoderNet.Learnables, avgGradientsEncoder, ...
            avgGradientsSquaredEncoder] = ...
            adamupdate(obj.EncoderNet.Learnables, encoderGrad, ...
            avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, ...
            learningRate);
        % end iteration
    end
    % end epoch
end

% shut down the parallel pool
delete(gcp('nocreate'));

%% set object property |Trained|
% after the network was successfully trained set the object property
% Trained to true
obj.Trained=true;

end
