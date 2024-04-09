classdef HyperparametersAED < handle
    % This class creates an object containing all the information
    % needed for training and setting up an object of the class
    % AutoencoderDeep.
    % Author :
    %    Anika Terbuch
    %
    % History :
    % \change{1.0.0}{07-Oct-2022}{Original}
    %  --------------------------------------------------
    % (c) 2022, Anika Terbuch
    %  Chair of Automation, University of Leoben, Austria
    %  email: automation@unileoben.ac.at
    %  url: automation.unileoben.ac.at
    %  --------------------------------------------------

    properties(SetAccess=private)
        % struct containing all the hyperparameters
        Hyperparameters
    end


    methods(Static)
        % this function creates a struct with default values for the
        % hyperparameters
        defaultStruct=setDefaultHyperparametersAED();
    end

    methods
        % constructor of the class
        function obj=HyperparametersAED
            % when creating a new object the hyperparameter vaues are
            % initialized with default values
            obj.Hyperparameters=HyperparametersAED.setDefaultHyperparametersAED();
        end

        % function to change the hyperparameters in the hyperparameter
        % struct with name value pairs as input
        setHyperparametersAED(obj, varargin);
    end

end
