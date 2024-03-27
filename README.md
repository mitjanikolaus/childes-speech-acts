# Speech Act Annotations
Classification of speech acts in child-caregiver conversations using CRFs, LSTMs and Transformers.
As recommended by the [CHAT transcription format](https://talkbank.org/manuals/CHAT.pdf), we use INCA-A as speech acts
annotation scheme.

This repository contains code accompanying the following papers:  

**[Large-scale study of speech acts' development in early childhood](https://ldr.lps.library.cmu.edu/article/532/galley/486/view/)**  
_Language Development Research (2022)_  
Mitja Nikolaus, Eliot Maes, Jeremy Auguste, Laurent Prévot and Abdellah Fourtassi


**[Modeling Speech Act Development in Early Childhood: The Role of Frequency and Linguistic Cues](https://escholarship.org/uc/item/10k7j8fr)**  
_In Proceedings of the 43nd Annual Meeting of the Cognitive Science Society. (2021)_  
Mitja Nikolaus, Eliot Maes and Abdellah Fourtassi


# Environment
An anaconda environment can be setup by using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate speech-acts
```

For processing data from childes-db, there are some additional requirements. First of all, you need to have R installed
on your system and install the `childesr` and `curl` R packages:
```
sudo apt-get install r-base
R
> install.packages("childesr")
> install.packages("curl")
```

Make sure the installations succeed. Afterwards, you can install the `childespy` python package:
```
pip install git+https://github.com/langcog/childespy.git
```

# Preprocessing data for supervised training of classifiers

Data for supervised training is taken from the [New England corpus](https://childes.talkbank.org/access/Eng-NA/NewEngland.html) of [CHILDES](https://childes.talkbank.org/access/).

1. Download the [New England Corpus data](https://childes.talkbank.org/data/Eng-NA/NewEngland.zip),
then extract and save it to `~/data/CHILDES/`.

2. Preprocess data
```
 python preprocess.py --drop-untagged
```

Alternatively, the data can be loaded from childes-db:
```
 python preprocess.py --childes-db --drop-untagged
```
  
# CRF
## Train CRF classifier

To train the CRF with the features as described in the paper:
```
python crf_train.py --use-pos --use-bi-grams --use-repetitions
```

## Test CRF classifier

Test the classifier on the same corpus:
```
python crf_test.py -checkpoint checkpoints/crf/
```

Test the classifier on the [Rollins corpus](https://childes.talkbank.org/access/Eng-NA/Rollins.html):
1. Use the steps described above to download the corpus and preprocess it.
2. Test the classifier on the corpus.
```
python crf_test.py --data data/rollins_preprocessed.p --checkpoint checkpoints/crf/
```
   
## Annotate using the CRF classifier

We provide a [trained checkpoint](checkpoint_full_train) of the CRF classifier. It can be applied to annotate new data.

The data should be stored in a CSV file, containing the following columns 
(see also [example.csv](examples/example.csv)).:
- `transcript_id`: the file id of the transcript
- `utterance_id`: unique id of the utterance within the transcript  
- `age`: child age in months
- `tokens`: a list of the tokens of the utterance
- `pos`: a list of part-of-speech tags for each token
- `speaker_code`: A value of `CHI` if the current speaker is the child, any other value is treated as adult speaker. 
 
An example for the creation of CSVs from
childes-db can be found in [preprocess_childes_db.py](preprocess_childes_db.py.).

Using `crf_annotate.py`, we can now annotate the speech acts for each utterance:
```
python crf_annotate.py --checkpoint checkpoint_full_train --data examples/example.csv --out data_annotated
```

An output CSV is stored to the indicated output file (`data_annotated/annotated.csv`). It contains an additional column
`speech_act` in which the predicted speech act is stored.

## Annotate data from childes-db using the CRF classifier

Preprocess data from childes-db (use the `--corpora` argument to preprocess only a subset of corpora):
```
python preprocess_childes_db.py --output-path utterances_childes_db.p
```

Annotate:
```
python crf_annotate.py --checkpoint checkpoint_full_train --data utterances_childes_db.p --out data_annotated
```
The output file contains an additional column `speech_act` in which the predicted speech act is stored.

# Neural Networks
(The neural networks should be trained on a GPU, see corresponding [sbatch scripts](sbatch-scripts).)

To run the neural networks you will also have to install Pytorch (>=1.4.0) in your environment.

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

