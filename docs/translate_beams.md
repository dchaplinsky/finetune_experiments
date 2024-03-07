# Usage Guide for translate_beams.py

## Links
1. Models list - https://github.com/Helsinki-NLP/UkrainianLT/blob/main/opus-mt-ukr-flores-devtest-big.md
2. Link to model(download) - https://object.pouta.csc.fi/Tatoeba-MT-models/eng-zle/opusTCv20210807+bt_transformer-big_2022-03-13.zip

## Running the code
1. Download the model from link 2
2. Convert model to compatible for ctranslate2 format:
```sh
ct2-opus-mt-converter --model_dir opusTCv20210807+bt_transformer-big_2022-03-13 --output_dir opusTCv20210807+bt_transformer-big_2022-03-13_ct2_model
```
3. Load data from Flores and store src:
```python
from datasets import load_dataset
import csv


dataset = load_dataset("facebook/flores", "eng_Latn-ukr_Cyrl")
dev = dataset["dev"]
devtest = dataset["devtest"]
dev.to_csv("flores-dev.csv")
devtest.to_csv("flores-devtest.csv")

eng = devtest["sentence_eng_Latn"]
def write_to_csv(list_of_emails):
    with open('flores-eng-devtest.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["eng_Latn-ukr_Cyrl"])
        writer.writeheader()
        for domain in list_of_emails:
            csvfile.write(domain + '\n')

write_to_csv(eng)

```
4. Preprocess eng src:
Exclude last 2 rows in else statement to exclude tokenization(as it is done in script)
```sh
./preprocess.sh eng ukr source.spm < flores-eng-devtest.csv > preprocessed_devtest.csv
```
1. Enjoy
```sh
python3 translate_beams.py --source-file-path=flores-devtest.csv --preprocessed-file-path=preprocessed_devtest.csv --target-file-path=target-opus.csv --translation-model-path=opus_ct2_model/ --tokenizer-model-path=./opus_ct2_model/source.spm --target-tokenizer-model-path=./opus_ct2_model/target.spm  --validation-field-name=sentence_ukr_Cyrl --source-field-name=sentence_eng_Latn  --src-prefix=">>ukr<<" --target-prefix=">>ukr<<" --beam-size=2
```
P.S. Postprocessing was implemented in script