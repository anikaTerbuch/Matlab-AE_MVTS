function [encoderGradients, decoderGradients] = gradientsRecErr(encoderNet, decoderNet, x)
% AED, AE, training
%
% Purpose : This function calculates the gradients of the reconstruction
% loss with respect to the learnables of the |encoderNet| and |decoderNet|
% after the samples in |x| are passed through.
% The reconstruction loss is calculated as the MSE between |x| and |xRec|
% which  is the output of the decoder - the reconstruction. 
%
% Syntax : gradientsRecErr(encoderNet, decoderNet, x)
%
% Input Parameters :
% -encoderNet: encoder of the AED
% -decoderNet: decoder of the AED
% -x: - data the networks should be evaluated on as DL-array with channels
%      'TSB'
%
%
% Return Parameters :
% -encoderGradients: gradients wrt. the learnables of the encoder evaluated 
% on the data |x|
% -decoderGradients: gradients wrt. the learnables of the decoder evaluated
% on the data |x|
%
% Description :
% The data passed to this function is encoded into the latent space. In the
% second space the reconstruction is obtained by decoding the data again
% back to the original domain using the decoder.
% The reconstruction error is calculated in a mean square sense. The
% gradients are calculated wrt. the reconstruction error and the encoder
% learnables and decoder learnables seperately.
%
% Author : 
%    Anika Terbuch
%
% History :
% \change{1.0}{19-Jan-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%


%% Encodings
% get the encodings (latent representation/ latent code z) of the samples
% in |x| 
% pass the samples in |x| through the encoder
latentRepresentation=forward(encoderNet,x);

%% Decodings
% get the reconstructed representation by passing the latent 
% representation z through the decoder network
reconstructedOutput=forward(decoderNet,latentRepresentation);

%% MSE reconstruction loss
% compute the reconstruction loss between |x| and |xRec| in a  mean
% square sense
squares=(x-reconstructedOutput).^2;
% calculate the reconstruction loss
recErr=sum(squares,[1,2,3]);


%% Gradient computation
% compute the gradients of the loss with respect to the learnables of the
% encoder (encoderNet.Learnables) and of the decoder
% (decoderNet.Learnables)

% gradients of learnables of both networks by calling the dlgradient function
[encoderGradients, decoderGradients]=dlgradient(recErr, ...
    encoderNet.Learnables, decoderNet.Learnables);
