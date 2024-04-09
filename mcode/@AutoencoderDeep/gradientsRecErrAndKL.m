function [encoderGradients, decoderGradients] = gradientsRecErrAndKL(encoderNet, decoderNet, x, weighingKL)
% AED, VAE, training
%
% Purpose : This function calculates the gradients of the reconstruction
% loss with respect to the learnables of the |encoderNet| and |decoderNet|
% after the samples in |x| are passed through.
% The reconstruction loss is calculated as the RMSE between |x| and |xRec|
% which  is the output of the decoder - the reconstruction. 
%
% Syntax :
%
% Input Parameters :
% -encoderNet: net which forms the encoder of the AutoenocoderDeep
% -decoderNet: net which forms the decoder of the AutoencoderDeep
% -x: dl-array with the samples for which the gradients should be evaluated
% -weightingKL: weighting factor for the KL-term in the cost function
%
%
% Return Parameters :
% -encoderGradients: gradients wrt.the learnables of the encoder
% -decoderGradients: gradients wrt. the learnables of the decoder
%
% Description :
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

% 3 Steps:
%% 1) Encoding
% obtain the encodings by calling the sampling function on the
% mini-batch that passes through the encoder network
% sampling step samples the mean and variance vectors to create the final
% encoding to be passed to the decoder network, because backpropagation
% through a random sampling operation is not possible, -> reparametrization
% trick
[z, zMean, zLogvar]=AutoencoderDeep.samplingVAE(encoderNet,x,0);


%% 2) ELBO Loss
% Obtain the loss by passing the encodings through the decoder network
% and calling the Elboloss function -> loss in VAEs -> ELBOloss

xPred=forward(decoderNet, z);
loss=AutoencoderDeep.ELBOloss(x, xPred, zMean, zLogvar,weighingKL);
loss=mean(abs(loss));



%% 3) Gradient computation
% compute the gradients of the loss with respect to the learnables of the
% encoder (encoderNet.Learnables) and of the decoder
% (decoderNet.Learnables)

% parameters of both networks by calling the dlgradient function
[decoderGradients, encoderGradients]=dlgradient(loss, ...
    decoderNet.Learnables, encoderNet.Learnables);
