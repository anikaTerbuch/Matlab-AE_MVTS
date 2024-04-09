function latentRepresentation = encodingAED(obj, data)
% AED, encoding, latent representation, encoder
%
% Purpose : Performs an encoding on the samples in |data| into the latent
% representation
%
% Syntax : latentRepresentation = encodingAED(obj, data)
%
% Input Parameters :
% - obj: AutoencoderDeep
% - data: data the encoding should be performed on (data type: dlarray)
%
% Return Parameters :
% - latentRepresentation: encoding of the data into the latent space
%
% Description :
% Applying the encoder to the data to get the encodings of the data in the
% latent space.
% 
%
% Author : 
%    Anika Terbuch
%
% History :
% \change{1.0}{21-Jan-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%

%% Encoder
% get the encoder
encoder=obj.EncoderNet;
% apply the encoder on the passed |data|
%% Distinction between VAE and AE
% get the type of the autoencoder
autoencoderType=obj.Hyperparameters.Hyperparameters.AutoencoderType;

%% Encoding if |obj.autoencoderType| is AE
if strcmp(autoencoderType,'AE')
latentRepresentation=predict(encoder,data);

%% Encoding if |obj.autoencoderType| is AE
elseif strcmp(autoencoderType,'VAE')
% sample the input |data| into the latent space 
%(encoding + reparametrization trick)
[latentRepresentation, ~, ~]=AutoencoderDeep.samplingVAE(encoder,data,obj.Trained);

end