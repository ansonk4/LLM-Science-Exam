import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
from __future__ import annotations

from collections.abc import Iterable

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader

# Helper function

def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)

# max sequence length for all-MiniLM-L6-v2
MAX_LENGTH = 384
# batch size for 
BATCH_SIZE = 16
# select top-K article
K = 3
# select top-L sentence
L = 20

# Relevant Title Retrieval

trn = pd.read_csv("/kaggle/input/raw-data/valid.csv")

SIM_MODEL = '/kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2'
model = SentenceTransformer(SIM_MODEL, device='cuda')
model.max_seq_length = MAX_LENGTH
model = model.half()

prompt_embeddings = model.encode(trn.prompt.values, batch_size=BATCH_SIZE, device=0, 
                                 show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
_ = gc.collect()

# title + first sentence
sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")
sentence_index = faiss.index_cpu_to_all_gpus(sentence_index)

## Get the top 5 pages that are likely to contain the topic of interest
search_score, search_index = sentence_index.search(prompt_embeddings, K)

search_score[:3], search_index[:3]

## Save memory - delete sentence_index since it is no longer necessary
del sentence_index
del prompt_embeddings
_ = gc.collect()
_ = libc.malloc_trim(0)

# Getting Sentences from the Relevant Titles

df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
                     columns=['id', 'file'])

## Get the article and associated file location using the index
wikipedia_file_data = []

for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    _df = df.loc[scr_idx].copy()
    _df['prompt_id'] = i
    wikipedia_file_data.append(_df)
wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

## Save memory - delete df since it is no longer necessary
del df
_ = gc.collect()
libc.malloc_trim(0)

wikipedia_file_data.head()

## Get the text data from the the parquet file 
wiki_text_data = []

for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
    # get all the id with the the parquet file = file
    _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
    
    _df = pd.read_parquet(f"/kaggle/input/wikipedia-20230701/{file}", columns=['id', 'text'])

    # get all the row of the parquet file where the id of that row is in _id
    _df_temp = _df[_df['id'].isin(_id)].copy()
    del _df
    _ = gc.collect()
    libc.malloc_trim(0)
    wiki_text_data.append(_df_temp)
wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
_ = gc.collect()

wiki_text_data.head()

## Parse documents into sentences
processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)
processed_wiki_text_data.head()

## Get embeddings of the wiki text data
wiki_data_embeddings = model.encode(processed_wiki_text_data.text,
                                    batch_size=BATCH_SIZE,
                                    device=0,
                                    show_progress_bar=True,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True)#.half()
wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
_ = gc.collect()

## Combine all answers
trn['answer_all'] = trn.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)

## Search using the prompt and answers to guide the search
trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']
trn.head()

question_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=0, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
question_embeddings = question_embeddings.detach().cpu().numpy()

# Extracting Matching Prompt-Sentence Pairs

## Parameter to determine how many relevant sentences to include
NUM_SENTENCES_INCLUDE = 20

## List containing just Context
contexts = []

for r in tqdm(trn.itertuples(), total=len(trn)):

    prompt_id = r.Index

    prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values
    
    if prompt_indices.shape[0] > 0:
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])

        context = ""
        
        ## Get the top matches
        ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
        for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
            context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + " "
        
    contexts.append(context)

trn['context'] = contexts
trn[["prompt", "context", "A", "B", "C", "D", "E"]].to_csv("./test_context.csv", index=False)