# Low Resource Arabic to English Machine Translation

## Project Overview
This project focuses on Low Resource Arabic to English Machine Translation.

## Dataset
- **Name**: IWSLT 2017
- **Hugging Face ID**: IWSLT/iwslt2017
- **Configuration**: iwslt2017-ar-en
- **Direction**: Arabic to English

## Week 1 Goal
Dataset download, cleaning, splitting, saving as text files, saving as CSV, saving as Parquet, BPE tokenization, statistics, plots, samples, and reports.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

## How to Run Notebooks
Open Jupyter Notebook and run the notebooks in the `notebooks/` directory in the following order:
1. `01_download_dataset.ipynb`
2. `02_clean_split_save_dataset.ipynb`
3. `03_statistics_and_plots.ipynb`
4. `04_tokenization_bpe.ipynb`

## Final Folder Structure
```text
low_resource_ar_en_mt/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_download_dataset.ipynb
│   ├── 02_clean_split_save_dataset.ipynb
│   ├── 03_statistics_and_plots.ipynb
│   └── 04_tokenization_bpe.ipynb
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── processed/
│   ├── samples/
│   ├── tokenized/
│   └── vocab/
│
├── plots/
│   ├── basic/
│   ├── lengths/
│   ├── quality_checks/
│   ├── vocabulary/
│   └── tokenization/
│
├── reports/
└── logs/
```

## Expected Outputs
The notebooks will download the IWSLT 2017 dataset, clean the Arabic and English pairs, restrict the training set to 50,000 pairs, train BPE tokenizers using SentencePiece, encode all splits into subword sequences, and generate various statistics and plots to understand the dataset's characteristics. Output files will include raw and clean text files, processed CSV and Parquet files, BPE-tokenized text files, SentencePiece model files, sample pairs, multiple PNG plots, and Markdown reports.

## Notes for Week 2
Week 2 will use the following generated files:
- `data/clean/train.ar` and `train.en` — clean plain text
- `data/clean/validation.ar`, `validation.en`, `test.ar`, `test.en`
- `data/tokenized/train.ar.bpe` and `train.en.bpe` — BPE-encoded text
- `data/tokenized/validation.*.bpe` and `test.*.bpe`
- `data/vocab/sp_ar.model` and `sp_en.model` — SentencePiece models for encoding/decoding
And may use these for analysis:
- `data/processed/train_clean.parquet`
- `data/processed/validation_clean.parquet`
- `data/processed/test_clean.parquet`
