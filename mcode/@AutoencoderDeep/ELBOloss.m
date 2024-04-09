function [elbo, elboLossTest] = ELBOloss(x, xPred, zMean, zLogvar, weightingKL)
% AED, ELBO, VAE, training
%
% Purpose : Calculates the weighted ELBO (weighting of the two terms of
% the cost function) for given data and given latent encodings.
% 
% Syntax : [elbo, elboLossTest] = ELBOloss(x, xPred, zMean, zLogvar, weightingKL)
%
%
% Input Parameters :
% -x: xData (original input)
% -xPred: predicted output of the VAE
% -zMean, zLogvar: learned parameters of the latent space
% -weightingKL: weighting factor of the KL-term in the cost function
%
% Return Parameters :
% -elbo: mean of the elbo-losses of all samples
% -elboLossTest: elbo loss for each sample seperately
% 
% Description : 
% The ELBOloss function takes the encodings of the means and 
% the variances returned by the sampling function and uses them to compute
% the ELBO loss.
% ELBOLoss = reconstruction Loss + weightingKL(-KLloss)
%
% Author :
%    Anika Terbuch, Stefan Herdy
%
% History :
% \change{1.0}{04-Nov-2021}{Original}
%
% --------------------------------------------------
% (c) 2021, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%

% quantification how close the decoder output is to the original input by using
% the mean-squared error and the KL divergence to calculate the divergence
% between the learned distribution and the prior.

squares = (xPred-x).^2;
reconstructionLoss  = sum(squares, [1,2,3]);
% negative Kullbach-Leibler divergence - measures difference between the two
% probability distributions.
KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
elbo=mean(reconstructionLoss+weightingKL*KL);
%elbo = mean(reconstructionLoss + KL);
% calculate the elboloss for every sample of x
elboLossTest=squares+KL;
% sum over all timesteps -> left: elboloss for every channel and every
% sample of x
elboLossTest=squeeze(sum(elboLossTest,3));
% sum over all channels of one sample -> left: elboloss of every sample of
% x
elboLossTest=squeeze(sum(elboLossTest,1));

end