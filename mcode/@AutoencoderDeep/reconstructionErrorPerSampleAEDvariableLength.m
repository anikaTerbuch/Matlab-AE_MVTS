function [reconstructionErrorPerSample, reconstructionErrorPerSampleNormalized, reconstructionError, reconstructionErrorPerChannel, reconstructionErrorPerChannelNormalized] = reconstructionErrorPerSampleAEDvariableLength(X,reconstructedX)
% AED, reconstruction error, reconstruction, evaluation
%
% Purpose : After applying encoding and decoding, this function calculates
% different variants of the reconstruction error using the 1-norm.
%
% Syntax : reconstructionErrorPerSampleAEDvariableLength(X,reconstructedX)
%
% Input Parameters :
% -X: input into the model on which the encoding and decoding was
% applied.
% -reconstructedX: time series after applying the encoding and decoding of
% a trained AutoencoderDeep to it
%
% Return Parameters :
% -reconstructionErrorPerSample - reconstruction error per pont
%  (scalar value per point) - summed up squared
% - reconstructionErrorPerSampleNormalized - reconstruction error
% normalized with the length of the time series (scalar value per point)
% reconstructionError: reconstruction error for each timestep, channel and
% point
% reconstructionErrorPerChannel: reconstruction error per pont and per
% channel (cell array - each sample )
% reconstructionErrorPerChannelNormalized: reconstruction error per point
% and per channel (cell array - each sample ) normalized by the length of the time
% series
%
% Description :
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{19-Mar-2021}{Original}
%
% --------------------------------------------------
% (c) 2021, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Loop over all the samples 
% variable length of reconstructed sequences! storing in 3-d-matrices not
% possible therefore each reconstructed sample is stored in a cell of a
% cell array.
    reconstructionErrorPerChannel={};
    reconstructionErrorPerChannelormalized={};
    reconstructionErrorPerChannel=[];
    reconstructionErrorPerChannelNormalized=[];
    % iterate over the number of samples
for i=1:length(X)
    % reset the variables
    ithX=[];
    reconstructionErrorSummed=[];

    % get the data of the i-th point
    ithX=X{i};
    ithReconstructedX=reconstructedX{i};

    % calculate the reconstruction error - 
    reconstructionError{i}=abs((ithReconstructedX-ithX));
    recErrIthChannel=squeeze(sum(reconstructionError{i},2));
    reconstructionErrorPerChannel(:,i)=recErrIthChannel;
    reconstructionErrorPerChannelNormalized(:,i)=reconstructionErrorPerChannel(:,i)/size(ithX,2);
    reconstructionErrorSummed=squeeze(sum(reconstructionErrorPerChannel(:,i),1));
    % normalize by the number of
    % timesteps because they are not constant
    reconstructionErrorPerSample(i)=reconstructionErrorSummed;
    reconstructionErrorPerSampleNormalized(i)=reconstructionErrorSummed/size(ithX,2);
end

