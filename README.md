# Speech Act Annotations
Classification of speech acts in child-caregiver conversations using CRFs, LSTMs and Transformers.
As recommended by the [CHAT transcription format](https://talkbank.org/manuals/CHAT.pdf), we use INCA-A as speech acts
annotation scheme.

This repository contains code accompanying the following papers:  

**Large-scale Study of Speech Acts' Development Using Automatic Labelling**  
_In Proceedings of the 43nd Annual Meeting of the Cognitive Science Society. (2021)_  
Mitja Nikolaus*, Juliette Maes*, Jeremy Auguste, Laurent Prévot and Abdellah Fourtassi (*Joint first authors)

**Modeling Speech Act Development in Early Childhood: The Role of Frequency and Linguistic Cues.**  
_In Proceedings of the 43nd Annual Meeting of the Cognitive Science Society. (2021)_  
Mitja Nikolaus, Juliette Maes and Abdellah Fourtassi


# Environment
An anaconda environment can be setup by using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate speech-acts
```

# Preprocessing data for supervised training of classifiers

Data for supervised training is taken from the [New England corpus](https://childes.talkbank.org/access/Eng-NA/NewEngland.html) of [CHILDES](https://childes.talkbank.org/access/) and then converted to XML:

1. Download the [New England Corpus data](https://childes.talkbank.org/data/Eng-NA/NewEngland.zip).
2. Convert the data using the [chatter java app](https://talkbank.org/software/chatter.html):
    ```
    $ java -cp chatter.jar org.talkbank.chatter.App [location_of_downloaded_corpus] -inputFormat cha -outputFormat xml -tree -outputDir java_out 
    ```
3. Preprocess data
    ```
    python preprocess.py --input-dir java_out/ --output-path data/new_england_preprocessed.p --drop-untagged
   ```
  
# CRF  
## Train CRF classifier

To train the CRF with the features as described in the paper:
```
python crf_train.py data/new_england_preprocessed.p --use-pos --use-bi-grams --use-repetitions
```

## Test CRF classifier

Test the classifier on the same corpus:
```
python crf_test.py data/new_england_preprocessed.p -m checkpoints/crf/ --use-pos --use-bi-grams --use-repetitions
```

Test the classifier on the [Rollins corpus](https://childes.talkbank.org/access/Eng-NA/Rollins.html):
1. Use the steps described above to download the corpus and preprocess it.
2. Test the classifier on the corpus.
   ```
   python crf_test.py data/rollins_preprocessed.p -m checkpoints/crf/ --use-pos --use-bi-grams --use-repetitions
   ```
   
## Apply the CRF classifier

We provide a [trained checkpoint](checkpoint) of the CRF classifier. It can be applied to annotate new data.

The data should be stored in a CSV file, containing the following columns 
(see also [example.csv](examples/example.csv)).:
- `file_id`: transcript ID  
- `child_id`: ID of the target child of the transcript
- `age_months`: child age in months
- `tokens`: A string containing the tokens of the utterance (separated by spaces)
- `pos`: part-of-speech tags for each token
- `speaker`: A value of `Target_Child` if the current speaker is the child, any other value is treated as adult speaker. 
 
An example for the creation of CSVs from
childes-db can be found in [preprocess_childes_db.py](preprocess_childes_db.py).

Using `crf_annotate.py`, we can now annotate the speech acts for each utterance:
```
python crf_annotate.py --model checkpoint --data examples/example.csv --out data_annotated
```

An output CSV is stored to the indicated directory (`data_annotated`). It contains an additional column `y_pred` 
in which the predicted speech act is stored.

# Neural Networks
(The neural networks should be trained on a GPU, see corresponding [sbatch scripts](sbatch-scripts).)

## LSTM classifier
### Training:
```
python nn_train.py --data data/new_england_preprocessed.p --model lstm --epochs 50 --out lstm/
```

### Testing:
```
python nn_test.py --model lstm --data data/new_england_preprocessed.p
```

## Transformer classifier (using BERT)
### Training:
```
python nn_train.py --data data/new-england_preprocessed.p --epochs 20 --model transformer --lr 0.00001 --out bert/
```

### Testing:
```
python nn_test.py --model bert --data data/new_england_preprocessed.p
```

# Collapsed force codes
The `collapsed_force_codes` branch contains code for analyses that utilize collapsed force codes, as described in:

**Modeling Speech Act Development in Early Childhood: The Role of Frequency and Linguistic Cues.**  
_In Proceedings of the 43nd Annual Meeting of the Cognitive Science Society. (2021)_  
Mitja Nikolaus, Juliette Maes and Abdellah Fourtassi

