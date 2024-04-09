function [reconstructionErrorPerSample, reconstructionErrorPerSampleNormalized, reconstructionError, reconstructionErrorPerChannel, reconstructionErrorPerChannelNormalized] = squaredReconstructionErrorPerSampleAEDvariableLength(X,reconstructedX)
%
% Purpose : After applying encoding and decoding, this function summs op
% the squared reconstruction error per sample 
%
% Syntax :
%
% Input Parameters :
% -X: input into the model on which the encoding and decoding was
% applied.
% -reconstructedX: time series after applying the encoding and decoding of
% a trained AutoencoderDeep to it
%
% Return Parameters :
% -reconstructionErrorPerSample - squared reconstruction error per point
%
% - reconstructionErrorPerSampleNormalized - squared reconstruction error
% normalized with the length of the time series (scalar value per point)
% reconstructionError: squared reconstruction error for each timestep, 
% channel and point
% reconstructionErrorPerChannel: squared reconstruction error per pont and
% per channel (cell array - each sample )
% reconstructionErrorPerChannelNormalized: squared reconstruction error per
% point and per channel (cell array - each sample ) normalized by the 
% length of the time series
%
% Description :
%
% Author :
%    Anika Terbuch, Gernot Steiner
%
% History :
% \change{1.0}{03-Mar-2022}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
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
for i=1:length(X)
    % reset the variables
    ithX=[];
    reconstructionErrorSummed=[];

    % get the data of the i-th point
    ithX=X{i};
    ithReconstructedX=reconstructedX{i};
    try
    % calculate the reconstruction error - 
    reconstructionError{i}=(ithReconstructedX-ithX).^2;
    recErrIthChannel=squeeze(sum(reconstructionError{i},2));
    reconstructionErrorPerChannel(:,i)=recErrIthChannel;
    reconstructionErrorPerChannelNormalized(:,i)=reconstructionErrorPerChannel(:,i)/size(ithX,2);
    reconstructionErrorSummed=squeeze(sum(reconstructionErrorPerChannel(:,i),1));
    % normalize by the number of
    % timesteps because they are not constant
    reconstructionErrorPerSample(i)=reconstructionErrorSummed;
    reconstructionErrorPerSampleNormalized(i)=reconstructionErrorSummed/size(ithX,2);
    catch
        % becaues there are samples with no data -> skip these samples
    end
end

