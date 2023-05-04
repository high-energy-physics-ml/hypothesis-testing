## Running jet-cnn code

The main files within this directory accomplish three main points: training the CNN on jet images, making predictions on type of jets within new data, and performing a simple hypothesis test to ascertain the significance level at which data can be said to be containing top jets. 

### Training the CNN and making predictions without bootstrapping

To train the CNN run 
```
python KerasCNN.py
```
This takes in data of QCD and top jet images, stored within `Data`, and outputs a trained CNN .h5 model file within a new directory called `model_cnn`.

To make predictions over new data run
```
python predictions.py
```
which uses the trained CNN model to find the probability of jet images from the testing data of being a top jet. These probabilities are saved within `cnn_outputs`.

### Training the CNN and making predictions with bootstrapping

One can also train the CNN with bootstrapping to account for uncertainties in the training process. To do this run
```
python KerasCNN_bootstrap.py
```
This trains the CNN over $N$ bootstraps and now, instead of saving a trained CNN model file, the predictions (as well as truth data and training scores) from each iteration of the bootstrapping are saved directly to `bootstrap_arrays`. There is therefore no need to run a seperate script for predictions (note that `predictions_from_bootstrap.py` is legacy experimental code and is no longer needed).

One should then run
```
python bootstrap_analysis.py
```
to find the average PDFs of the predictions from bootstrapping which are then saved to `cnn_outputs`. This script also returns plots which analyse the results from bootstrapping.

### Running the Log-Likelihood Ratio simple hypothesis test

To perform the hypothesis test run
```
python jet_llr.py
```
This reads in the predictions from `cnn_outputs` (which can be produced with or without bootstrapping). It then performs a simple hypothesis test with data that contains only QCD events, or data that contains QCD and top events (mixed with appropriate cross-sections). To do this it samples a number of events from the full reference PDFs for the QCD only and QCD + top mixed cases. The Log-Likelihood Ratio (LLR) is then calculated using the reference PDFs but with evenets actually sampled from either the QCD or mixed case. This is done for many toy experiments to build a distribution of LLRs from which the significance level $\alpha$ and the equivalent number of standard deviations $n_\sigma$ can be found. This is done for a range of detector luminosities and the results are saved to `arrays`.

### Viewing results

The results can be plotted by running (inside the `results` directory)
```
python naive_pdf_cut_vs_llr_quick_results.py
```
The script loads in the arrays previously saved (as well as files containing the probabilities of an event being a top or qcd, as found in `cnn_outputs`, which will be used for calculating the standard significances for comparision). It first calculates the standard significances through obtaining the numbers of top and qcd events that would, on average, be obtained in an experiment at a given luminosity. It does this by considering the PDFs so that cuts can be made on it, if desired. This is done for $S/\sqrt{B}, $S/\sqrt{S+B}$ and the Azimov significance, although they are all essentially equivalent here. Finally the results of $\alpha$ and $n_\sigma$ produced by `jet_llr.py` are loaded in and plotted alongside the standard significances.

### Additional code

The file `gaussian_smear.py` demonstrates applying a Gaussian smearing to the jet images to simulate noise. This file is not used for anything else and is for producing example plots only.

The file `bootstrap_analysis.py` can be run to view metrics of the bootstrapping performance such as the confidence interval of the accuracy scores from each boostrap and the overall PDFs. This file is otherwise not used for anything else.
