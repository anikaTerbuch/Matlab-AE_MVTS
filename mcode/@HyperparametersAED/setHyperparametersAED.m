function setHyperparametersAED(obj, varargin)
% hyperparameters, AED, input checking, defining valid inputs
%
% Purpose : With this function the hyperparameter values of the struct of
% hyperparameters can be changed.
%
% Syntax : setHyperparametersAED(obj, varargin)
%
% Input Parameters :
% - obj: AutoencoderDeep
% - vargin: name-value pairs of the fieldnames of the struct of
% hyperparamters and the corresponding values.
%
% Return Parameters :
%
% Description : This function performs input checking based on a
% pre-defined set of possible input values (data type or categorical value)
% and assigns the hypeprarameter to the new value when the checks were 
% sucessfull.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{4.0}{14-Mar-2022}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%
% This function can only be executed when the network has not been trained.
% The hyperparameters influence the network's architecture and settings
% during training.

% %% Check if Autoencoder is trained
% if obj.Trained
%     error(['This AutoencoderDeep was trained, consequently the ' ...
%         'hyperparameters can no longer be changed'])
% end

%% Check if the number of passed arguments is even - name value pairs
nrInputs = numel( varargin );

if mod(nrInputs,2)==1
    error(['Unexpected number of inputs. ' ...
        'The inputs need to be a name-value pairs.'])
end

%% Define struct of valid values for input assertion
%
% valid values for the hyperparmaeters of the struct which have categorical
% values
%
% the field names of this struct are constructed prefixing the fieldnames
% of the hyperparameter struct with valid and postfixing them with a s
% example: fieldname in hyperparameter-struct = LatentDim
% fieldname in this struct is validLatentDims
validVals=struct;
% numeric parameters - positive integer
validVals.validLatentDims="posInteger";
validVals.validNumberEpochs="posInteger";
validVals.validNumberFeatures="posInteger";
validVals.validMiniBatchSizes="posInteger";
% positive numeric value
validVals.validWeightingKLs="posNumeric";
%
validVals.validLearningRates="posNumericMax1";
% categorical values
validVals.validExecutionEnvironments={'cpu','gpu','auto'};
validVals.validAutoencoderTypes={'VAE','AE'};
% array of categorical values - define two 'valid Vals' - one for datatye
% one for categorical values
validVals.validLayersEncoders="categoricalArray";
validVals.validLayersDecoders="categoricalArray";
%
validVals.validLayersEncodersTypes={'FC','LSTM','Bi-LSTM'};
validVals.validLayersDecodersTypes={'FC','LSTM','Bi-LSTM'};
%
% output transfer function
validVals.validOutputTransferFunctions={'sigmoid','tanh','none'};

% arrays of positive integers, number of neurons per layers needs to be a
% positive integer; the size of each layer corresponds to an entry in the
% array
validVals.validNeuronsEncoders="posIntegerArray";
validVals.validNeuronsDecoders="posIntegerArray";


%% Devide into names and values
% odd entries are names, even entries are the corresponding values
names = varargin(1:2:nrInputs-1);
vals = varargin(2:2:nrInputs);

%% Assertion of the hyperparameters which should be changed
%
% get the struct of hyperparameters out of the object
S=obj.Hyperparameters;
% check if the passed values have valid values for the variables. The
% possible inputs (types or categorical values) are defined in the
% struct |validVals|

% iterate over the passed name-value pairs
for v=1:length(names)
    % get the current name-value pair
    fieldname=names{v};
    fieldvalue=vals{v};
    % check if the passed fieldname |fieldname| is a valid fieldname of the
    % hyperparameter struct
    if isfield(S,fieldname)
        % construct the string which is expected to be the fieldname in the
        % struct |validVals| if the |fieldname| is a valid value.
        strName=char(strcat('valid',(fieldname),'s'));
        % checks if for the fieldname a set of valid values is defined
        % -> if the current fieldname is a field in the struct of valid
        % values
        if isfield(validVals,strName)
            % check if in the struct of valid vals the valid values of the
            % field are specified in a cell
            % categorical values
            if iscell(validVals.(strName))
                % checks if the return array of valid strings is not empty
                % checks if the passed value matches one of the specified
                % values
                if ~any(validatestring(fieldvalue,validVals.(strName)))
                    error(fieldvalue)
                end
                % the valid values are not specified in a cell -> data type
                % specified -> check if the value should be a logical
                % input or numeric input
            else
                % pull out the value specified in the struct of valid
                % variables -> indicates of which type the variable should
                % be
                varType=validVals.(strName);
                if varType=="posNumericMax1"
                    % check if passed fieldvalue is numeric and positive
                    % between 0 and 1
                    if ~isnumeric(fieldvalue) || fieldvalue > 1 || fieldvalue < 0
                        error(['This hyperparameter can only take ' ...
                            'numeric values between 0 and 1.'])
                    end

                    if (fieldvalue >1 || fieldvalue < 0)
                        error(['The hyperarameter LearningRate needs to be' ...
                            'between 0 and 1'])
                    end
                elseif varType=="logical"
                    % check if the passed fieldvalue is logical
                    if ~islogical(fieldvalue)
                        error(['This hyperparameter can only take ' ...
                            'logical values.'])
                    end
                elseif varType=="posInteger"
                    % check if the passed fieldvalue is an integer
                    if ~(fieldvalue == floor(fieldvalue)) || fieldvalue < 0
                        error(['This hyperparameter can only take ' ...
                            'positive integer values.'])
                    end
                elseif varType=="posNumeric"
                    if ~isnumeric(fieldvalue) || fieldvalue < 0
                        error(['This hyperparameter can only take ' ...
                            'positive numeric values.'])
                    end
                elseif varType=="categoricalArray"
                    if ~iscell(fieldvalue)
                        errorMsg=strcat(fieldname,': This hyperparameter can only take cell-arrays as its values');
                        error(errorMsg)
                    end
                    categoricalVals=char(strcat(strName,'Types'));
                    % iterate over the length of the passed array - check
                    % for each entry if passed entry is a valid categorical
                    % value
                    for i=1:length(fieldvalue)
                        % checks if the return array of valid strings is not empty
                        % checks if the passed value matches one of the specified
                        % values
                        ithfieldvalue=string(fieldvalue{i});
                        if ~any(validatestring(ithfieldvalue,validVals.(categoricalVals)))
                            error(ithfieldvalue)
                        end
                    end
                elseif varType=="posIntegerArray"
                    % check if each entry of the passed array is numeric
                    % and positive
                    for i=1:length(fieldvalue)
                        if ~(isnumeric(fieldvalue(i)))
                            error(strcat('The hyperparameter',(fieldname) ...
                                ,'can only take numeric entries.'))
                        end
                        if ~(abs(fieldvalue(i))==floor(fieldvalue(i)))
                            error(strcat(['Non integer entry in array' ...
                                '  ',fieldname,'.']))
                        end
                    end
                else
                    error(strcat('No valid value for field '," ",(fieldname)))
                end
            end
        end
    else
        % throw an error if the field entered is not a valid fieldname
        error(strcat('Entered fieldname '," ", fieldname, [' does' ...
            ' not match any hyperparameter.']))
    end
    %% Assigning
    % set the parameter (fieldname) to the new value fieldvalue
    S.(fieldname)=fieldvalue;

end
% before final-assigning the hyperarameter struct to the hyperparameter of
% the object - check if the dimensions of the EncoderTypes and
% EncoderNeurons and DecoderTypes and DecoderNeurons match.
if ~(length(S.LayersEncoder)==length(S.NeuronsEncoder))
    error(['Dimensions of LayersEncoder and NeuronsEncoder do not match.' ...
        'Arrays need to have the same length.'])
end

if ~(length(S.LayersDecoder)==length(S.NeuronsDecoder))
    error(['Dimensions of LayersDecoder and NeuronsDecoder do not match.' ...
        'Arrays need to have the same length.'])
end

%% weightingKL - conditional field - only needed when autoencoderType is a VAE
% if the AutoencoderType is an autoencoder and the field for weighting the
% KL-divergence in the cost function exists, remove this field
if S.AutoencoderType=="AE" && isfield(S,"WeightingKL")
    S = rmfield(S,"WeightingKL");
end


% if the AutoencoderType is a variational autoencoder and it does not has a
% field for the weighting for the KL divergence set it to a default value
if S.AutoencoderType=="VAE" && ~isfield(S,"WeightingKL")
    S.WeightingKL=defaultWeigthingKL;
end


% re-assign the struct of hyperparameters to the object
obj.Hyperparameters=S;

end


