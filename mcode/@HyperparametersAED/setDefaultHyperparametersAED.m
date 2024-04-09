function defaultStruct=setDefaultHyperparametersAED()
% default Hyperparameters, AED
%
% Purpose : This function sets the hyperparameters of objects of the class
% AutoencoderDeep to default values. The hyperarameter struct is a property
% of the class HyperparametersAED
%
% Syntax : setDefaultHyperparametersAED(obj)
%
% Input Parameters :
% obj: HyperparametersAED
%
% Return Parameters :
% defaultStruct: a struct containing the default values for the
% hyperparameters
%
% Description :
% The default values for the hyperparameters of the AutoencoderDeep are
% defined. The struct of hyperparameters is created and the default values 
% are assigned to it. The created struct is assigned as the hyperparameter
% struct of the passed object.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{14-Jan-2022}{Original}
% \change{2.0}{09-Feb-2022}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%
% Struct which contains the default hyperparameters
defaultStruct=struct();

%% define the default values

% number of neurons in the layers of the encoder
defaultNeuronsEncoder=[50,20];
% number of neurons in the  layers of the decoder
defaultNeuronsDecoder=[30];
% dimension of the latent space
defaultLatentDim=2;
% number of epochs (runs through the training set) during training
defaultNumberEpoch=10;
% number of features passed to the network -> automatically adjusted when 
% the training data is passed to the network when calling the training 
% function trainAED()
defaultNumberFeature=1;
% initial learning rate used for the training with the adamupdate
% (adaptive momentum estimation - adaptive learning rate)
defaultLearningRate=0.05;
% size of a mini-batch during training - number of samples which is passed
% through the network before a gradient step is done
defaultMiniBatchSize=15;
% environment on which the learning is executed - auto - it is checked if
% hardware for gpu-learning is available if not the learning is executed on
% the cpu
defaultExecutionenvirionment='auto';
% which types of layers are used in the encoder
defaultLayersEncoder={'FC','LSTM'};
% which types of layers are used in the decoder
defaultLayersDecoder={'LSTM'};
% type of latent space
defaultAutoencoderType='VAE';
% output transfer function - function which is applied on the outputs of
% the decoder
defaultOutputTransferFunction='none';

% conditional hyperparameter - only added to the struct when the
% AutoencoderType=='VAE'
% lambda - weigthing of the Kullback-Leibler-term of the cost function of
% the variatonal autoencoder
defaultWeightingKL=1;

% create the fields of the struct and assign the pre-defined default
% values
defaultStruct.AutoencoderType=defaultAutoencoderType;
defaultStruct.LayersEncoder=defaultLayersEncoder;
defaultStruct.LayersDecoder=defaultLayersDecoder;
defaultStruct.NeuronsEncoder=defaultNeuronsEncoder;
defaultStruct.NeuronsDecoder=defaultNeuronsDecoder;
defaultStruct.LatentDim=defaultLatentDim;
defaultStruct.NumberEpoch=defaultNumberEpoch;
defaultStruct.NumberFeature=defaultNumberFeature;
defaultStruct.LearningRate=defaultLearningRate;
defaultStruct.MiniBatchSize=defaultMiniBatchSize;
defaultStruct.ExecutionEnvironment=defaultExecutionenvirionment;
defaultStruct.WeightingKL=defaultWeightingKL;
defaultStruct.OutputTransferFunction=defaultOutputTransferFunction;











