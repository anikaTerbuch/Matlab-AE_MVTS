# Matlab-AE_MVTS
Generic Deep Autoencoder for Time-Series 

This toolbox enables the simple implementation of different deep autoencoder. The primary focus is on multi-channel time-series analysis.
Each autoencoder consists of two, possibly deep, neural networks - the encoder and the decoder.
The following layers can be combined and stacked to form the neural networks which form the encoder and decoder:
- LSTM (Long-short term memory layers),
- Bi-LSTM (Bi-directional long-short term memory layers),
- FC with ReLU (Fully connected layers followed by a rectified linear unit).

There are two types of autoencoders available:
- AE (autoencoder)
- VAE (variational autoencoder)
The autoencoders can be easily parametrized using hyperparameters.

This toolbox is also mirrored to [MatlabFileExchange](https://de.mathworks.com/matlabcentral/fileexchange/111110-generic-deep-autoencoder-for-time-series).


-------------------------------------------------------------------------------------------
# Publishing information

This code can also be considered as supplemental Material to the Paper:
"Hybrid Machine Learning for Anomaly Detection in Industrial Time-Series
Measurement Data" 
by: Anika Terbuch, Paul O'Leary and Peter Auer
Mai 2022

The paper can be found in the paper [Hybrid Machine Learning for Anomaly Detecion in Industrial Time Series (https://ieeexplore.ieee.org/document/9806663)

# Example Plot
This plot shows the input and output of a multivariate time-series to a variational autoencoder built with this framework

![AutoencoderDeep](https://github.com/anikaTerbuch/Matlab-AE_MVTS/assets/58983404/ca537240-d2e0-4c59-b1a0-7abe78e1ce5b)

-------------------------------------------------------------------------------------------
# Class descriptions
The toolbox includes the following two classes:
1) AutoencoderDeep:  generic framework for creating autoencoders suitable for multivariate time-series data
2) HyperparametersAED: the class used to parametrize objects of the class AutoencoderDeep.
-------------------------------------------------------------------------------------------
For easier and more convinient use the classes are wrapped into functions:
The following functions are provided:
-trainAutoencoderDeep: trains an autoencoder using the data provided.
-predictAutoencoderDeep: returns an prediction of the autoencoder passed to the function on the data provided to the function.


## AutoencoderDeep

ad 1) 

The class AutoencoderDeep contains the following functions:
- decodingAED:         	decodes the latent representation  back into the original domain.

- ELBOloss:             calculates the Evidence Lower Bound (ELBO) of given data, given
                        latent encoding and the weighting factor of the two terms of the loss function.

- encodingAED:          performs an encoding into the latent space of a trained
                        AutoencoderDeep on the samples provided to the function.

- gradientsRecErr:      loss function of the AE - minimize the reconstruction loss
                        evaluates the encoder and decoder on the passed data and calculates
                        the gradients of the learnable parameters of the network.

- gradientsRecErrAndKL: loss function of the VAE - maximize the evidence lower bound
                        evaluates the encoder and decoder on the passed data and calculates
                        the gradients of the learnables of the network.

- layerArray2dlnetwork: converts a layer array to a dl-network.

- reconstructionAED:    performs the reoncustriction (encoding followed by decoding) on the 
                        given data with a trained AutoencodeDeep.
						 
- reconstructionErrorPerSampleAEDvariableLength: given the reconstructed signal and the 
                                                 real signal the reconstruction error (1-
                                                 norm) normalized by the varying length
						                         of the time series is calculated.
												
- samplingVAE:          performs the encoding into the latent space followed by the 
                        reparametrization trick used in the calculation of latent
                        representations when using VAEs.
						 


-setUpAndTrainAED:      creates the encoder and decoder networks and trains them accoding
                        to the properties specified in the hyperparameter struct.
						 
-setUpDecoderAED:	sets up a neural network that forms the decoder of the network with the
                    layer-types and number of neurons specified in the hyperparameter
                    struct.
						
-setUpEncoderAED:   sets up a neural network that forms the encoder of the network with
                    the layer-types and number of neurons specified in the 
                    hyperparameter struct.						 
					 
-setUpEncoderDecoderAED: function calls to functions used for setting up the encoder and 
                         decoder networks.

-squaredReconstructionErrorPerSampleAEDvariableLength: given the reconstructed signal and
                                                       the real signal the reconstruction
                                                       error (2-norm) normalized by the
                                                       varying length of the time series is
                                                       calculated.
													   
-trainingLoopAED:		training loop for training the encoder and decoder simultaneously
                         based on ADAM, training can be performed using CPU-units or GPU-
                         units.
						
-varSeqLen2dlarray:     	creates mini-batches of dl-arrays with minimum resampling. 		

## HyperparametersAED
ad 2) 

The class HyperparametersAED contains the following functions:
-setDefaultHyperparametersAED: initializes the parameters to default values at instantiation of an object		 

-setHyperparametersAED: can be used to change the hyperparameters of the AutoencoderDeep by 
                        passing the hyperaramter names to change and their values 
                        ('name value pairs') to this function.

								
-------------------------------------------------------------------------------------------

This derived class is tailored to handle data shown in
https://de.mathworks.com/help/deeplearning/ug/time-series-anomaly-detection-using-deep-learning.html
https://de.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html and
https://de.mathworks.com/help/deeplearning/ug/sequence-to-one-regression-using-deep-learning.html?searchHighlight=WaveformData&s_tid=srchtitle_WaveformData_3

The following impelementation was created using Matlab 2022a and 2022b.
