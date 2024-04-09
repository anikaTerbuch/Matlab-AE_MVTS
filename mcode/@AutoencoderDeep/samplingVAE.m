function [latentRepresentation, zMean, zLogvar] = samplingVAE(encoderNet, data, trained)
% AED, VAE, training
% Purpose : This function performs an encoding to a given data and returns
% the mean and the variance of the VAE. This function can be used during
% training of an AutoencoderDeep and to get a latent encoding for a already
% trained AutoencoderDeep.
%
% Syntax : samplingVAE(encoderNet, data, trained)
%
% Input Parameters :
% obj: AutoencoderDeep
% data: data which should be sampled to the latent space
% trained: boolean which indicates if the encoder was already trained
%   
% Return Parameters :
% zMean, zLogvar: parameters of the distribution 
% latentRepresentation; latentRepresentation = epsilon .* sigma + zMean
% 
%
% Author :
%    Stefan Herdy, Anika Terbuch
%
% History :
%
%
% --------------------------------------------------
% (c) 2020, Stefan Herdy, Anika Tebuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: stefan.herdy@stud.unileoben.ac.at
% --------------------------------------------------

%% Distinction if function is called during training or not
% object alredy trained, function applied on trained encoder to get
% encoding in the latent space
if(trained)
    % calling function predict: does not change learnables when the data is
    % passed through the encoder network
    latentPrediction=predict(encoderNet,data);
else
    % function called during training - data should influence the
    % learnables of the encoder network
    latentPrediction=forward(encoderNet,data);
end
% the representation in the latent space is described by the mean and the
% variance
% d - dimension of the latent space - number of latent variables
d = size(latentPrediction,1)/2;
% first half of the predicted values in the latent space corresponds to the
% means
zMean=latentPrediction(1:d,:,:);
% second half of the predicted values in the latent space corresponds to
% the log of the variances
% network outputs log-variance because network needs to be constrained to
% learn positive variance.
zLogvar=latentPrediction(1+d:end,:,:);
%% Reparametrization trick
% dimension of the encodings in the latent space - dlarray
sz=size(zMean);
% create random numbers in the size of sz - random portion to make backprop
% through a random node possible
epsilon=randn(sz);
% calculate the standard deviation when the log(variance) is given
sigma=exp(.5 * zLogvar);
% latent representation after reparametriziaton trick
latentRepresentation=epsilon .* sigma + zMean;
% convertion to dlarray with desired channels
latentRepresentation=dlarray(latentRepresentation, 'CBT');

end

