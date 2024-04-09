function dlNetwork= layerArray2dlnetwork(layerArray)
% AED, layer array, dl-network, layer graph
%
% Purpose : This function converts an array of layers to a dlnetwork.
%
% Syntax : dlNetwork= layerArray2dlnetwork(layerArray)
%
% Input Parameters :
% - layerArray: array of layers
%
% Return Parameters :
% - dlNetwork: a dlnetwork with the layers specified in the layerArray.
%
% Description :
% A layer graph is created and the passed layers of the layer array are
% added to it. The created layer graph is then casted to a dlNetwork.
%
% Author : 
%    Anika Terbuch
%
% History :
% \change{1.0}{03-Feb-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%
%% convert the layers to a layer graph and add the layers
lgraph=layerGraph;
lgraph=addLayers(lgraph,layerArray);

%% convert the layer graph to a dlnetwork
dlNetwork = dlnetwork(lgraph);