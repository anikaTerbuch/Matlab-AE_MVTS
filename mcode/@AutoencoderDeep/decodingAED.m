function reconstructedOutput = decodingAED(obj,latentRepresentation)
% AED, decoding, latent representation, decoder
%
% Purpose : Decodes the latent representation  back into the original
% domain.
%
% Syntax : reconstructedOutput = decodingAED(obj,latentRepresentation)
%
% Input Parameters :
% - obj: AutoencoderDeep()
% - latentRepresentation: latent representation which should be decoded
% with the decoder back into the original domain.
%
% Return Parameters :
% - reconstructedOutput: reconstructed output after applying the decoder to
% the latent representation.
%
% Description : This function decodes the latent variables back to the
% original domain. There is a distinction made: if the autoencoder was not
% trained before and if it was trained before.  This is due to the fact,
% that during training the weights are still changing based on the input to
% a neural network. When the training is completed, the weights are not
% changed anymore.
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
%% Decoding
% get the decoder
decoder=obj.DecoderNet;
% apply the decoder on the encoded latent representation
% distinction if AutoencoderDeep was already trained or not
if obj.Trained
    % use the function predict on trained networks
    % reconstructedOutput=sigmoid(predict(decoder,latentRepresentation));
    reconstructedOutput=predict(decoder,latentRepresentation);
else
    % use the function forward during training
    % reconstructedOutput=sigmoid(forward(decoder,latentRepresentation));
    reconstructedOutput=forward(decoder,latentRepresentation);
end