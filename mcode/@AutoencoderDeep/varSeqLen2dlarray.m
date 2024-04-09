function [cellOfDlArrays, numIterations] = varSeqLen2dlarray(XData, miniBatchSize,executionEnvironment)
% mini-batch, resampling, sorting by time series length
%
% Purpose : This function converts a cell array which contains matrices -
% each associated with data from a file - to dlArrays which are resampled
% to the shortest sequence length of the samples which form a miniBatch.
% Consequently, the dl-arrays do not have all the same sequence length.
% The samples in XData are sorted according to their length and then form
% mini-batches (which are dlarrays) of size |batchSize|
%
% Syntax : varSeqLen2dlarray(XData, miniBatchSize,executionEnvironment)
%
% Input Parameters :
% - XData, sequenceLength ->
% outputs of function generateVAEInputLSTMvaryingLength
% miniBatchSize - desired mini-batch-size (number of samples that form a
% batch in a DL-array for training)
%
% Return Parameters :
%
% Description :
% Sorts the passed cells according to their lenght. Devides them in
% mini-batches. Each mini-batch gets resampled to a common length (length
% of the shortest time-series of each batch) and casted to a dlarray with
% the dimensions 'TCB'.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{10-Feb-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Sorting
% the time series should be sorted according to the time-series length
[sortedLength,idx] = sort( cellfun( 'size', XData,2 ));

%% determine if input data consists of time series of diffrent length
% sort only if the input data consists of time series of different length
if length(unique(sortedLength))>1
    XData=XData(idx);
end

%% splitting into mini-batches
numIterations=floor(length(sortedLength)/miniBatchSize);
% possible: residual
res=mod(length(sortedLength),miniBatchSize);

allDLbatch={};
for iteration=1:numIterations
    % get the indices of the training samples which are used in this
    % iteration
    % lowest index used in this iteration
    idxL=(iteration-1)*miniBatchSize+1;
    % highest index used this iteration
    idxU=iteration*miniBatchSize;
    % indices of the samples used in this iteration
    idxIteration=idxL:idxU;
    % get the lengths of the sequences of this mini-batch
    batchLength=sortedLength(idxIteration);
    % get the minimal sequence length of this mini-batch - sorted ascending
    % - first entry - downsampling all samples that form a batch to the size of the
    % shortest sequence
    minLen=batchLength(1);
    % resample the samples into this length

    % iterte over the data of each sample which is part of the mini-batch
    bData=[];
    for b=1:miniBatchSize
        % get the b-th sample
        bSamp=XData{idxIteration(b)};
        bSampR=[];
        % resample each channel of the b-th sample
        for c=1:size(bSamp,1)
            % get the data of the current channel
            currChan=bSamp(c,:);

            %% Resampling:
            resizedChanVal=imresize(currChan,[1,minLen],'bicubic');
            bSampR(c,:)=resizedChanVal';
        end
        bData(b,:,:)=bSampR;

    end
    %% converting into a dl-array
    % convert the data of the current batch to a dl-array with the
    % number of maxLen timesteps
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        cellOfDlArrays{iteration}=gpuArray(dlarray(bData,'BCT'));
    else    
        cellOfDlArrays{iteration}=dlarray(bData,'BCT');
    end

end



