function setUpEncoderDecoderAED(obj)
% AED, encoder, decoder
%
% Purpose : This function executes the functions for setting up the encoder
% and decoder.
%
% Syntax : setUpEncoderDecoderAED(obj)
%
% Input Parameters :
% -obj: AutoencoderDeep
%
% Return Parameters :
%
% Description :
%
% Author : 
%    Anika Terbuch
%
% History :
% \change{1.0}{16-Feb-2022}{Original}
% \change{2.0}{09-Mar-2022}
%
% --------------------------------------------------
% (c) 2022, Anika Terbuch
% Chair of Automation, University of Leoben, Austria
% email: automation@unileoben.ac.at
% url: automation.unileoben.ac.at
% --------------------------------------------------
%
%% Encoders and decoders in the time domain
obj.setUpEncoderAED()
obj.setUpDecoderAED()

end