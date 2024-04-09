function [latentRepresentation, reconstructedOutput,failedRec, input]=reconstructionAED(obj,data)
% AED, trained, evaluating data
%
% Purpose : This function performs the reconstruction of the data given in
% |data| with a trained AutoencoderDeep.
%
% Syntax : reconstructionAED(obj,data)
%
% Input Parameters :
% -obj: AutoencoderDeep
% -data: data (cell array of time-series)
%
% Return Parameters :
% -latentRepresentation: latent representation of each sample in |data|
% -reconstructedOutput: reconstructed output of each sample in |data|
% -failedRec: array containing the indices of the samples where the
% reconstruction failed
% -input: returns the data passed to the function
%
% Description :
% Each cell of data is firstly converted into a dl-array.
% For each sample in the passed data set an encoding followed by a decoding
% is performed by passing the data firstly through the encoder network to
% get the latent encoding and this then gets reconstructed using the
% decoder.
%
% Author :
%    Anika Terbuch
%
% History :
% \change{1.0}{16-Feb-2022}{Original}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%%
failedRec=[];
% not for each sample data available - skip points without data
cnt=1;

% iterate over the cells in data - convert cell into dlarray - perform
% encoding followed by decoding
for i=1:length(data)
    % convert sample in dl-array with 'batch' size 1
    ithdl=dlarray(data{i},'CTB');

    %% Encoding
    % apply the trained encoder to the data to get the latent
    try
        ithLatentRepr=encodingAED(obj, ithdl);
        % convert dlarray to cell of cell array
        ithLatentReprArray=squeeze(extractdata(ithLatentRepr));
        % if the latent representation is a gpu array convert to conventional
        % matlab array
        if(isa(ithLatentReprArray,'gpuArray'))
            ithLatentReprArray=gather(ithLatentReprArray);
        end

        latentRepresentation{cnt}=ithLatentReprArray;


        %% Decoding
        % apply the trained decoder to the data to get the reconstructed output of
        % the AutoencoderDeep
        ithRec=decodingAED(obj,ithLatentRepr);

        % convert from dl-array to cell of cell array
        reconstructedOutputArray=squeeze(extractdata(ithRec));
        % check if the reconstructedOutputArray is a gpu array - if yes convert
        % to a conventional matlab array

        if(isa(reconstructedOutputArray,'gpuArray'))
            reconstructedOutputArray=gather(reconstructedOutputArray);
        end

        reconstructedOutput{cnt}=reconstructedOutputArray;

        input{cnt}=data{i};
        cnt=cnt+1;

    catch
        % if a reconstruction fails, return the index of the data that
        % could not be reconstructed
        failedRec(length(failedRec)+1)=i;
    end

end