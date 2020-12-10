# Speech Act Annotations
Repository for classification of speech acts in child-caregiver conversations using CRFs, LSTMs and Transformers.

As recommended by the [CHAT transcription format](https://talkbank.org/manuals/CHAT.pdf), we use INCA-A as speech acts
annotation scheme.

# Requirements
Listed in `environment.yml`

# Preprocessing data for supervised training of classifiers

Data for supervised training is taken from the [New England corpus](https://childes.talkbank.org/access/Eng-NA/NewEngland.html) of [CHILDES](https://childes.talkbank.org/access/) and then converted to XML:

1. Download the [New England Corpus data](https://childes.talkbank.org/data/Eng-NA/NewEngland.zip).
2. Convert the data using the [chatter java app](https://talkbank.org/software/chatter.html):
    ```
    $ java -cp chatter.jar org.talkbank.chatter.App [location_of_downloaded_corpus] -inputFormat cha -outputFormat xml -tree -outputDir java_out 
    ```
3. Preprocess data
    ```
    python preprocess.py --input-path java_out/ --output-path data/new_england_preprocessed.p
   ```
   
# Train CRF classifier

```
python crf_train.py data/new_england_preprocessed.p [optional_feature_args]
```

# Test CRF classifier


```
python crf_test.py data/new_england_preprocessed.p -m checkpoints/crf/
```
