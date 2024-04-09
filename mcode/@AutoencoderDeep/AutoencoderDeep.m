classdef (InferiorClasses = {?dlnetwork}) AutoencoderDeep < handle
    % AutoencoderDeep is an object-oriented implementation of a deep
    % Autoencoder which can be applied to multi-channel time-series data
    % sets. The hyperparameters are defined as seperate object with the
    % class @HyperparametersAED
    % All the functions associated with this class are postfixed with "AED".
    %
    % Author :
    %    Anika Terbuch
    %
    % History :
    % \change{2.0.0}{30-Mar-2022}{Original}
    % \change{3.0.0}{09-Jan-2023}
    %
    % cite as:
    % @INPROCEEDINGS{Terbuch2022,
    % AUTHOR="Anika Terbuch and Paul O'Leary and Peter Auer",
    % TITLE="Hybrid Machine Learning for Anomaly Detection in Industrial
    % {Time-Series} Measurement Data",
    % BOOKTITLE="2022 IEEE International Instrumentation and 
    % Measurement Technology Conference (I2MTC) (I2MTC 2022)",
    % ADDRESS="Ottawa, Canada",
    % DAYS=16,
    % MONTH=may,
    % YEAR=2022,
    % }
    %
    %  --------------------------------------------------
    % (c) 2023, Anika Terbuch
    %  Chair of Automation, University of Leoben, Austria
    %  email: automation@unileoben.ac.at
    %  url: automation.unileoben.ac.at
    %  --------------------------------------------------
    %
    %-----------------------properties----------------------------------------
    % constant properties
    properties(Constant)
        % versioning
        Version = '3.0.0'
        Description={['This object was created with the class' ...
            ' AutoencoderDeep. This is an object-oriented implementation' ...
            ' of autoencoders which can consist out of fully-connected, LSTM' ...
            'and Bi-LSTM layers.']}
    end

    % private properties - can be viewed from outside but not changed
    properties(SetAccess=protected)
        % boolean which indicates if the object was trained or not
        Trained
        % object of the class HyperparametersAED
        Hyperparameters
    end

    % hidden properties
    properties(Hidden, SetAccess=protected)
        % Networks:
        % dlnetwork network which forms the encoder
        EncoderNet
        % dlnetwork network which forms the decoder
        DecoderNet
        % reference to the data-path of the data used for training
        DataTrain
    end


    %-----------------------methods----------------------------------------
    % public methods
    methods
        function obj=AutoencoderDeep(hyperparameters)
            % AutoencoderDeep constructor of the class
            % constructor of the class AutoencoderDeep
            
            % the property trained keeps track if the object was allready
            % trained (property value is chnged to true). At initialization
            % it is set to false
            obj.Trained=false;
            obj.Hyperparameters=hyperparameters;
        end

    end

    % signatures for methods implemented in seperate function files
    % public methods
    methods

    end

    % abstract methods which require data ingestion tailored to the use


    methods(Hidden)
        % this functions are used to perform hyperparameter optimization
        % when calling this methods the data ingestion is done prior to
        % them and not at the instanciation of each object to reduce the
        % runtime of the hyperparameter optimization

        % method for training the AED with beforehand data ingestion
        setUpAndTrainAED(obj,data)
        % encoding and decoding for inputs of varying length with
        % berforehand data ingestion
        [latentRepresentation, reconstructedOutput,failedRec, originalInput]=...
            reconstructionAED(obj,data)
    end

    % protected methods - internal use not accessable by the user
    methods(Access=protected)
        % sets up an encoder and decoder according to the settings in the
        % hyperparameter struct
        setUpEncoderDecoderAED(obj);
        % creates the neural network which forms the encoder
        % according to the hyperparameters
        setUpEncoderAED(obj);
        % creates the neural network which forms the decoder
        % according to the hyperparameters
        setUpDecoderAED(obj);
        % calculating the gradient - different loss functions
        trainingLoopAED(obj,data,filenames, gradientFunction);
        % encoding
        latentRepresentation=encodingAED(obj,data);
        % decoding
        reconstructedOutput=decodingAED(obj,latentRepresentation);
    end

    methods(Static, Access=protected)
        % converts layer array into dlnetwork
        dlNetwork= layerArray2dlnetwork(layerArray);
        % cost function for the autoencoder
        [encoderGradients, decoderGradients] = ...
            gradientsRecErr(encoderNet, decoderNet, x);
        % cost function for the variational autoencoder
        [encoderGradients, decoderGradients] = ...
            gradientsRecErrAndKL(encoderNet, decoderNet, x, weighingKL)
        % performs the reparametriziaton trick (VAE)
        [latentRepresentation, zMean, zLogvar] = ...
            samplingVAE(encoderNet, data, trained)
        % calculates the ELBO-loss (minimizing training objective of the
        % VAE)
        [elbo, elboLossTest] = ELBOloss(x, xPred, zMean, zLogvar, weightingKL)
        % converts the number of cells which form a mini-batch to a
        % dl-array
        [cellOfDlArrays, numIterations] = ...
            varSeqLen2dlarray(XData, miniBatchSize, executionEnvironment)
    end

    methods(Static, Hidden)
        % performance evaluation - 1-norm of the reconstruction error
        [reconstructionErrorPerSample, reconstructionErrorPerSampleNormalized, ...
            reconstructionError, reconstructionErrorPerChannel,...
            reconstructionErrorPerChannelNormalized] = ...
            reconstructionErrorPerSampleAEDvariableLength(X,reconstructedX)

       % performance evaluation - 2-norm of the reconstruction error
        [reconstructionErrorPerSample, reconstructionErrorPerSampleNormalized,...
            reconstructionError, reconstructionErrorPerChannel,...
            reconstructionErrorPerChannelNormalized] = ...
            squaredReconstructionErrorPerSampleAEDvariableLength(originalInput,reconstructedOutput)

    end
end

