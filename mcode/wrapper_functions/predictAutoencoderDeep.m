function varargout=predictAutoencoderDeep(varargin)
% <keywords>
%
% Purpose : This function is used to compute the predictedOutput using the
% trained Autoencoder and the data passed with this function.
%
% Syntax:
% [reconstructedOutput]=predictAutoencoderDeep(cellOfData, trainedAutoencoder)
% optional output arguments:
% [reconstructedOutput,latentRepresentation]=predictAutoencoderDeep(__)
% [reconstructedOutput,latentRepresentation,reconstructionErrorPerSampleNormalized]=predictAutoencoderDeep(__)
% [reconstructedOutput,latentRepresentation,reconstructionErrorPerSampleNormalized,reconstructionErrorPerChannelNormalized]=predictAutoencoderDeep(__)
% [reconstructedOutput,latentRepresentation,reconstructionErrorPerSampleNormalized,reconstructionErrorPerChannelNormalized,failedIndex]=predictAutoencoderDeep(__)
% [reconstructedOutput,latentRepresentation,reconstructionErrorPerSampleNormalized,reconstructionErrorPerChannelNormalized,failedIndex,originalInput]=predictAutoencoderDeep(__)
% 
%
% Input Parameters :
% -cellOfData - data  the prediction should be performed on
% -trainedAutoencoder - autoencoder which was trained before 
%
% Return Parameters :
% - reconstructedOutput
% - latentRepresentation (optional)
% - reconstructionErrorPerSampleNormalized:
%   squared reconstruction error per sample normalized by the length of
%   the original time-series (optional)
% - reconstructionErrorPerChannelNormalized: 
%   squared reconstruction error per channel normalized by the length of 
%   the original time-series (optional)
% - indexFailed: index of samples where the reconstruction failed
%   (optional)
% - originalInput (optional)
%
% Description :
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{28-Oct-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Distinction based on the number of input arguments passed
% determine the number of inputs passed
numIn=length(varargin);

switch numIn
    case {0,1}
        % no input passed -> no allowed case; at least data needs to
        % be passed that a prediction can be made
        error(['This function requires 2 inputs: a trained autoencoder,' ...
            'a cell of data the prediction should be performed on'])

    case 2
        % two inputs passed - first input assumed to be data; second input
        % assumed to be a trained autoencoder
        cellOfData=varargin{1};
        trainedAutoencoder=varargin{2};
        % check if passed object is a trained autoencoder
        assert(isequal(class(trainedAutoencoder), 'AutoencoderDeep'), ...
            'The passed object needs to be an object of the class AutoencoderDeep.');
        % check if the autoencoder is trained
        assert(trainedAutoencoder.Trained == true,['The Autoencoder is ' ...
            'not trained. Training needs to be performed before ' ...
            'predictions can be done.']);
        % assert if the passed data is a cell
        assert(iscell(cellOfData),['The passed data is invalid; ' ...
            'needs to be stored in a cell']);

    otherwise
        error(['Function executed with wrong number of function-inputs.' ...
            'This function can be executed with 1 or 2 inputs.'])
end




%% Initializations 
reconstructedOutput={};
latentRepresentation={};
%% Data preprocessing

% save the original input into a cell array
originalInput=cellOfData;
% get the latent representation and the reconstruction of the data in
% |data|
[latentRepresentation, reconstructedOutput, failedRec, ...
    originalInput]=trainedAutoencoder.reconstructionAED(cellOfData);

% calculate the reconstruction error between the reconstructed output
% and the originalInput
[~, reconstructionErrorPerSampleNormalized, ~, ~, ...
    reconstructionErrorPerChannelNormalized] = AutoencoderDeep.squaredReconstructionErrorPerSampleAEDvariableLength(originalInput,reconstructedOutput);

%% Variable number of outputs
% determine how many output-arguments are requested
numOut=nargout;

assert(numOut>0,'This function cannot be executed without outputs.' )
% set the outputs based on the number of requested outputs 
i=1;
while i<(numOut+1)
    if i==1
        
        varargout{1}=reconstructedOutput';
    elseif i==2
        varargout{2}= latentRepresentation';

    elseif i==3
        varargout{3}=reconstructionErrorPerSampleNormalized';

    elseif i==4
        varargout{4}=reconstructionErrorPerChannelNormalized';
    elseif i==5
        varargout{5}=failedRec;
    elseif i==6
        varargout{6}=originalInput';
    else 
        warning(['Too many output arguments specified. Some of them may' ...
            ' not be set.'])
    end
    i=i+1;
end



end



