## Running eft-dnn code

The main files within this directory accomplish three main points: training the DNN on SM and SMEFT kinematic data, making predictions on the type of events within new data, and performing a simple hypothesis test to ascertain the significance level at which data can be said to be containing SMEFT signals. 

### Training the DNN and making predictions without bootstrapping

To train the DNN run 
```
python eft_KerasCNN.py
```
This takes in data of SM and SMEFT kinematic event data, stored within `Data`, and outputs a trained DNN .h5 model file within a new directory called `model_dnn`.

To make predictions over new data run
```
python eft_dnn_predictions.py
```
which uses the trained DNN model to find the probability of events within the testing data of being a SMEFT signal event. These probabilities are saved to a .txt file within the main directory (note that they can then be moved to `dnn_outputs` manually - this proccess should be automated in the future).

### Training the DNN and making predictions with bootstrapping

One can also train the DNN with bootstrapping to account for uncertainties in the training process. To do this run
```
python eft_dnn_predictions_bootstrap.py
```
This trains the DNN over $N$ bootstraps and now, instead of saving a trained DNN model file, outputs the predictions from each iteration of the bootstrapping are saved to one .txt file within the main directory (note that they can then be moved to `dnn_outputs` manually - this proccess should be automated in the future). Also note that we do not find the average PDF within an analysis file as we did for the jet-cnn (as we mainly did that for analysing the bootstrapping process), instead this whole .txt file is read in by `eft_dnn_lrr.py` which will compute the PDF directly.

### Running the Log-Likelihood Ratio simple hypothesis test

To perform the hypothesis test run
```
python eft_dnn_llr.py
```
This reads in the predictions from `dnn_outputs` (which can be produced with or without bootstrapping). It then performs a simple hypothesis test with data that contains only SM background events, or data that contains SM background and SMEFT signal events (mixed with appropriate cross-sections). To do this it samples a number of events from the full reference PDFs for the SM only and SM + SMEFT mixed cases. The Log-Likelihood Ratio (LLR) is then calculated using the reference PDFs but with evenets actually sampled from either the SM or mixed case. This is done for many toy experiments to build a distribution of LLRs from which the significance level $\alpha$ and the equivalent number of standard deviations $n_\sigma$ can be found. This is done for a range of detector luminosities and the results are saved to `arrays`.

### Viewing results

The results can be plotted by running (inside the `results` directory)
```
python naive_pdf_cut_vs_llr.py
```
The script loads in the arrays previously saved (as well as files containing the probabilities of an event being a SM event or a SMEFT signal event, as found in `dnn_outputs`, which will be used for calculating the standard significances for comparision). It first calculates the standard significances through obtaining the numbers of SM and SMEFT signal events that would, on average, be obtained in an experiment at a given luminosity. It does this by considering the PDFs so that cuts can be made on it, if desired. This is done for $S/\sqrt{B}, $S/\sqrt{S+B}$ and the Azimov significance, although they are all essentially equivalent here. Finally the results of $\alpha$ and $n_\sigma$ produced by `eft_dnn_llr.py` are loaded in and plotted alongside the standard significances.

