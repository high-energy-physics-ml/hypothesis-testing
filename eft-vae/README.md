## Running eft-vae code

The main files within this directory accomplish three main points: training the VAE on SM and SMEFT kinematic data, making predictions on the type of events within new data, and performing a general hypothesis test to ascertain the discovery significance for data which contains SMEFT signals. 

### Training the VAE and finding the reconstruction error without bootstrapping

To train the VAE run 
```
python eft_vae_predictions.py
```
This takes in data of SM kinematic event data, stored within `Data`, and trains the VAE on this data, which it saves as a .h5 file within `models`. Once the VAE is trained the script will then compute the reconstruction error (as the mean squared error) for new data containing only SM events or new data containing both SM events with some SMEFT signal events. 

Note that unlike with the supervised training scripts, the predictions are made within the same script here for simplicity's sake. The predictions (i.e. the reconstruction error) are saved as .txt files within `vae_outputs`. 

### Training the VAE and finding the reconstruction error with bootstrapping

One can also train the VAE with bootstrapping to account for uncertainties in the training process. To do this run
```
python eft_vae_predictions_bootstrap.py
```
This trains the VAE over $N$ bootstraps and now, instead of saving a trained VAE model file, outputs the predictions from each iteration of the bootstrapping are saved to one .txt file within `vae_outputs`.

### Running the Log-Likelihood Ratio general hypothesis test

To perform the general hypothesis test run
```
python eft_vae_llr_general.py
```
This reads in the predictions from `vae_outputs` (which can be produced with or without bootstrapping). It then performs a generalised hypotheis test by comparing the reconstruction error PDFs that would be obtained with data that contains only SM background events, or data that contains SM background and SMEFT signal events (mixed with appropriate cross-sections). Note that for our generalised hypothesis test we cannot take the signal PDF to be known a-priori, so we take the average LLR value obtained from simulated toy experiments, in a manner similar to before, but then calculate the $p$-value (which is converted also into $Z$) from this value using the (half)-$\chi^2_1$ distribution. The results from this are saved within `arrays`.

### Viewing results

The results can be plotted by running (inside the `results` directory)
```
python naive_pdf_cut_vs_llr.py
```
The script loads in the arrays previously saved (as well as files containing the probabilities of an event being a SM event or a SMEFT signal event, as found in `vae_outputs`, which will be used for calculating the standard significances for comparision). It first calculates the standard significances through obtaining the numbers of SM and SMEFT signal events that would, on average, be obtained in an experiment at a given luminosity. It does this by considering the PDFs so that cuts can be made on it, if desired. This is done for $S/\sqrt{B}, $S/\sqrt{S+B}$ and the Azimov significance, although they are all essentially equivalent here. Finally the results of $Z$ produced by `eft_vae_llr_general.py` (as well as the results for $n_\sigma$ from the supervised dnn for comparision) are loaded in and plotted alongside the standard significances.

### Running a Log-Likelihood Ratio simple hypothesis test (additional code)

One can also perform a simple hypothesis test using the VAE. Doing so kind of defeats the purpose of using a VAE trained without knowledge of the SMEFT signal, but is totally possible to do since we have truth information of the SMEFT. We do not include results from this in the paper, but it can be run here for comparison. To do this run
```
python eft_vae_llr_general.py
```
This reads in the predictions from `vae_outputs` (which can be produced with or without bootstrapping). It then performs a simple hypothesis test with data that contains only SM background events, or data that contains SM background and SMEFT signal events (mixed with appropriate cross-sections). To do this it samples a number of events from the full reference reconstruction error PDFs for the SM only and SM + SMEFT mixed cases. The Log-Likelihood Ratio (LLR) is then calculated using the reference PDFs but with evenets actually sampled from either the SM or mixed case. This is done for many toy experiments to build a distribution of LLRs from which the significance level $\alpha$ and the equivalent number of standard deviations $n_\sigma$ can be found. This is done for a range of detector luminosities and the results are saved to `arrays`.

