function setUpAndTrainAED(obj,data)
% train, AED, set up network
%
% Purpose : This function warps the functions for setting up the encoder
% and decoder network and training them.
%
% Syntax : setUpAndTrainAED(obj,data)
%
% Input Parameters :
% obj: AutoencoderDeep
% data: data the AutoencoderDeep should be trained on, each sample should
% be stored in a cell of a cell array
%
% Return Parameters : 
%
% Description :
% This function firstly adjustes the number of features according to the
% data the AutoencoderDeep should be trained on.
% After that it sets up the encoder and decoder networks according to the
% hyperparameters specified.
% Finally, it trains the networks on the data according to the specified
% loss function by the type of autoencoder used.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{16-Feb-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Set the number of features
% set the number of features to the number of input channels of the
% dlarray|data|
obj.Hyperparameters.setHyperparametersAED('NumberFeature',size(data{1},1))

%% Set up encoder and decoder
obj.setUpEncoderDecoderAED();


%% Training of the encoder and decoder
% distinction between AE and VAE -> different loss functions

AutoencoderType=obj.Hyperparameters.Hyperparameters.AutoencoderType;
% train an AutoencoderDeep of the AutoencodeType "Autoencoder"
if strcmp(AutoencoderType, 'AE')
        obj.trainingLoopAED(data,@AutoencoderDeep.gradientsRecErr);
% train an AutoencoderDeep of the AutoencodeType "Variational Autoencoder"

elseif strcmp(AutoencoderType,'VAE')
    obj.trainingLoopAED(data,@AutoencoderDeep.gradientsRecErrAndKL);
end

