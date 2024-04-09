function varargout=trainAutoencoderDeep(varargin)
% AED, trainAED, wrapper
%
% Purpose : This function trains an autoencoder using the data provided.
% 
% This function wraps the training procedure of an object
% of the class autoencoder deep.
%
% Syntax :
% [trainedAutoencoder]=trainAutoencoderDeep(cellOfData)
% [trainedAutoencoder]=trainAutoencoderDeep(cellOfData, hyperparameters)
%
% Input Parameters :
% -cellOfData: training data stored in a cell of cells
% -hyperparameters: object of class |hyperparametersAED| containing the
% hyperparameters of the autoencoder (optional). More information of the
% hyperparameters can be found in the Introduction.pdf
%
% Return Parameters :
% - trainedAutoencoder: object of the class AutoencoderDeep which was
% trained using the training data provided in cellOfData
%
% Description :
% In this function an object of the class AutoencoderDeep according to the
% hyperparameters specified in the hyperparameters. More information on the
% available hyperparameter-options can be found in the file
% Introduction. If no hyperparameters are specified then an autoencoder
% with default hyperparameters is created.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{20-Dec-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Variable input number
% determine the number of inputs passed
numIn=length(varargin);

% check how many inputs are passed to the function
switch numIn
    case 0
        % no input passed to function -> invalid case
        error(['To train the Autoencoder pass the data as a cell array ' ...
            'to this function.'])
    case 1
        % the first passed argument is expected to be the training data
        % stored in a cell
        cellOfData=varargin{1};
        % one input passed to function -> assumed to be data
        % check if it is a cell array
        assert(iscell(cellOfData),['The passed data is invalid; ' ...
            'needs to be stored in a cell'])
        % create a default hyperparameter-struct
        hyperparameters=HyperparametersAED();
        
    case 2
        % data and hyperparameter-struct were passed
        % check the two passed arguments
        cellOfData=varargin{1};
        hyperparameters=varargin{2};
        % check the type of passed inputs

        % the first passed argument is expected to be the training data
        % stored in a cell
        assert(iscell(cellOfData),['The passed data is invalid; ' ...
            'needs to be stored in a cell'])
        % the second passed argument should be of the class hyperparameters
        assert(isequal(class(hyperparameters), 'HyperparametersAED'), ...
            'The passed hyperparameters are invalid')
    otherwise
        error(['Wrong number of inputs passed; the number of inputs' ...
            'should be between 0 and 2'])
end

% check the dimensions of the data - each cell needs to have the dimensions
% [numberOfFeatures x numberOfObservations]
[nrows,~] = cellfun(@size,cellOfData);

% check if the first dimension is consistent over all cells
assert(length(unique(nrows)),['Check the dimensions of the passed data.' ...
    'Each cell needs to contain the same number of features.']);


% initializing an autoencoder with the hyperparameters
ae=AutoencoderDeep(hyperparameters);
% display the hyperparameters
ae.Hyperparameters.Hyperparameters
% training the autoencoder |ae| on the data contained in |cellOfData|
ae.setUpAndTrainAED(cellOfData);
% check if the training was successful
assert(ae.Trained==1,['Something went wrong and the training of ' ...
    'the autoencoder failed.']);

%% Variable number of outputs
% determine how many output-arguments are requested
numOut=nargout;

% check how many output parameters are requested.
switch numOut
    case 0
      error('No return parameter specified, the trained autoencoder cannot be saved.')
    case 1
        % if an output is requested, return the trained autoencoder
        varargout{1}=ae;
    otherwise
        % if the number of requested variables is higher than one, return
        % the trained autoencoder and a warning.
        warning(['This function returns one output parameter. ' ...
            'Some of your variables may not be set.'])
        varargout{1}=ae;
end

end
