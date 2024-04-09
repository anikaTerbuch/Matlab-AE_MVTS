function setUpEncoderAED(obj)
% AED, encoder, layers, layer array
%
% Purpose : This function creates the encoder of the AutoencoderDeep
% according to the hyperparameters secified in the hyperparameter struct
%
% Syntax : setUpEncoderAED(obj)
%
% Input Parameters :
% -obj: AutoencoderDeep
%
% Return Parameters :
%
% Description :
% This function sets up the network of the data type dlnetwork, which forms
% the encoder of the AutoencoderDeep, according to the hyperparameters.
%
% Each encoder has the following structure:
% - sequence input layer - dimension corresponds to the number of input
%   features
%  -layers with specific functionallity specified in the hyperparameters
%    available layers: fully connected  FC (followed by ReLu), LSTM, Bi-LSTM
%  -fully connected layer - output dimension is dependant on the
%     latent dimension.
%
%  The layers are stored in a graph, this graph is converted to a so called
%  layer graph and further to a dl-network - which is matlabs ML-structure
%  to implement customized neural networks.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{09-Mar-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Distinction between the Autoencoder Types
aeType=obj.Hyperparameters.Hyperparameters.AutoencoderType;
% distinguish the autoencoder types
switch aeType
    % outDimLatent: dimension of the data that is outputted to the latent space
    case 'AE'
        % dimenison of the output to the latent space is the latent dimension
        outDimLatent=obj.Hyperparameters.Hyperparameters.LatentDim;

    case 'VAE'
        % dimenison of the output to the latent space is twice the latent
        % dimension - using an VAE: modelling probability distributions in the
        % latent space
        outDimLatent=obj.Hyperparameters.Hyperparameters.LatentDim*2;
end
%% Setting up the layers
%number of LSTM layers that should be added to the encoder-network
numberLayers=length(obj.Hyperparameters.Hyperparameters.NeuronsEncoder);
% first layer of network: sequence input layer
layersEncoder(1)=sequenceInputLayer(obj.Hyperparameters.Hyperparameters.NumberFeature, ...
    'Name','inputLayerEncoder');

% add the layers secified in |Hyperarameter.LayersEncoder|
for i=1:numberLayers
    % check which layer-type the i-th layer should have
    layerType=obj.Hyperparameters.Hyperparameters.LayersEncoder{i};

    % distinguish the layer types
    switch layerType
        %LSTM layer
        case 'LSTM'
        layername=strcat('lstmE',string(i));
        layersEncoder(length(layersEncoder)+1)=lstmLayer(obj.Hyperparameters. ...
            Hyperparameters.NeuronsEncoder(i),'Name', (layername), ...
            'InputWeightsInitializer','he','RecurrentWeightsInitializer','he');

        % Bi-LSTM layer
        case 'Bi-LSTM'
        layername=strcat('bilstmE',string(i));
        layersEncoder(length(layersEncoder)+1)=bilstmLayer(obj.Hyperparameters. ...
            Hyperparameters.NeuronsEncoder(i),'Name', (layername), ...
            'InputWeightsInitializer','he','RecurrentWeightsInitializer','he');

        % Fully connected layer
        case 'FC'
        layername=strcat('FCE',string(i));
        layersEncoder(length(layersEncoder)+1)=fullyConnectedLayer(obj.Hyperparameters. ...
            Hyperparameters.NeuronsEncoder(i),'Name', (layername),'WeightsInitializer','he');
        layersEncoder(length(layersEncoder)+1)=reluLayer;
     % end switch
    end
% end for
end

% add a fully connected layer as last layer of the encoder
layersEncoder(length(layersEncoder)+1)=fullyConnectedLayer(outDimLatent, ...
    'Name','fcE','WeightsInitializer','he');

%% Converting the array of layers to a dlnetwork
encoderNet=AutoencoderDeep.layerArray2dlnetwork(layersEncoder);
% assign the encoder network to the object
obj.EncoderNet=encoderNet;

end


