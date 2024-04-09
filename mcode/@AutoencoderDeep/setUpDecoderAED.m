function setUpDecoderAED(obj)
% AED, decoder, layers, layer array
%
% Purpose : This function creates the decoder of the AutoencoderDeep
% according to the hyperparameters secified in the hyperparameter struct.
%
% Syntax : setUpDecoderAED(obj)
%
% Input Parameters :
% - obj: AutoencoderDeep
%
% Return Parameters :
%
% Description :
% This function sets up the network of the data type dlnetwork which forms
% the decoder of the AutoencoderDeep according to the hyperparameters.
% Each decoder has the following structure:
%  - sequence input layer - dimension is the dimension of the latent space
%  - layers with specific functionallity specified in the hyperparameters
%    available layers: fully connected followed by ReLu, LSTM, Bi-LSTM
%  - fully connected layer - output dimension corresponds to the number of
%    features in the input data.
%  - optional: a layer which implements a non-linearity and defines the 
% output range of the machine learning model (e.g. sigmoid, tanh). 
% The type of  non-linearity used is defined in the hyperparameter struct.
% The layers are stored in a graph, this graph is converted to a so called
% layer graph and further to a dl-network - which is matlabs ML-structure
% to implement customized neural networks.
% The layers of the network are initialized using the 'He'-initializer.
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
%%
%number of specified layers that should be added to the decoder-network
numberLayers=length(obj.Hyperparameters.Hyperparameters.NeuronsDecoder);

% first layer of network: sequence input layer
layersDecoder(1)=sequenceInputLayer(obj.Hyperparameters.Hyperparameters.LatentDim, ...
    'Name','inputLayerDecoder');

% add the layers secified in |Hyperarameter.LayersDecoder|
for i=1:numberLayers
    % check which layer-type the i-th layer should have
    layerType=obj.Hyperparameters.Hyperparameters.LayersDecoder{i};

    % distinguish the layer types
    switch layerType
        %LSTM layer
        case 'LSTM'
            layername=strcat('lstmD',string(i));
            layersDecoder(length(layersDecoder)+1)=lstmLayer(obj.Hyperparameters. ...
                Hyperparameters.NeuronsDecoder(i),'Name', (layername), ...
                'InputWeightsInitializer','he','RecurrentWeightsInitializer','he');

         % Bi-LSTM layer
        case 'Bi-LSTM'
            layername=strcat('bilstmD',string(i));
            layersDecoder(length(layersDecoder)+1)=bilstmLayer(obj.Hyperparameters. ...
                Hyperparameters.NeuronsDecoder(i),'Name', (layername), ...
                'InputWeightsInitializer','he','RecurrentWeightsInitializer','he');

        % Fully connected layer
        case 'FC'
            layername=strcat('FCD',string(i));
            layersDecoder(length(layersDecoder)+1)=fullyConnectedLayer(obj.Hyperparameters. ...
                Hyperparameters.NeuronsEncoder(i),'Name', (layername), ...
                'WeightsInitializer','he');
            layersDecoder(length(layersDecoder)+1)=reluLayer;
    % end switch        
    end
% end for    
end

% add a fully connected layer as last layer of the decoder
layersDecoder(length(layersDecoder)+1)=fullyConnectedLayer(obj.Hyperparameters. ...
  Hyperparameters.NumberFeature,'Name','fcD');

% optional: get the type of output transfer function
outputTF=obj.Hyperparameters.Hyperparameters.OutputTransferFunction;

if strcmp(outputTF,"sigmoid")
    layersDecoder(length(layersDecoder)+1)=sigmoidLayer;
elseif strcmp(outputTF,"tanh")
    layersDecoder(length(layersDecoder)+1)=tanhLayer;
end
% if outputTF=="none" -> don't add an output function as last layer



%% Convert the array of layers to a dlnetwork
decoderNet=AutoencoderDeep.layerArray2dlnetwork(layersDecoder);
% assign the encoder network to the object
obj.DecoderNet=decoderNet;
end
