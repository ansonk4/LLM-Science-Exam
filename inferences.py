# Perform prediction with the MC question with context generated by rag.py, two predictions are made, one with deberta-v3 on all question and the second with mistral-7b on the 12.5% hardest questions.
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Union
import gc, torch

test_df_2 = pd.read_csv("test_context_2.csv")

# We'll create a dictionary to convert option names (A, B, C, D, E) into indices and back again
options = 'ABCDE'
indices = list(range(5))
option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}

def preprocess(example):
    first_sentence = [ "[CLS] " + example['context'] ] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                  max_length=1024, add_special_tokens=False)
    return tokenized_example

model_dir = "/kaggle/input/merge-model/model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right", truncation_side="right")

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        return batch

def inference(test_df, device: str):
    test_predictions = []
    
    model = AutoModelForMultipleChoice.from_pretrained(model_dir).to(device).eval()
    
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(preprocess, remove_columns=list(test_df.keys()))
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
        test_predictions.append(outputs.logits.cpu().detach())
        
    test_predictions = torch.cat(test_predictions)
    
    del model, test_dataset, tokenized_test_dataset, data_collator, test_dataloader
    
    return test_predictions

def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


### first prediction 
test_file = "/kaggle/input/faiss-k5-l20/valid.csv"
test_label_file = "/kaggle/input/faiss-k5-l20/valid.csv"

out = inference(test_df_2, "cuda")

test_predictions = out

predictions_as_ids = np.argsort(-test_predictions)[:,:3]
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = [
    ' '.join(row) for row in predictions_as_answer_letters
]

test_df = pd.read_csv(test_file)
test_df['prediction'] = predictions_as_string
sub = test_df[['id', 'prediction']]
sub.to_csv("prediction1.csv", index=False)

try:
    train_df = pd.read_csv(test_label_file)
    m = MAP_at_3(test_df.prediction.values, train_df.answer.values)
    print( 'CV MAP@3 =',m )
except Exception as e:
    print(e)


### second prediction for the 12.5% hardest questions

gc.collect()
torch.cuda.empty_cache()

test_predictions = out
prec = torch.nn.functional.softmax(test_predictions)
top1 = torch.max(prec, dim=1).values
worst_question_ids = torch.topk(-top1, int(len(top1) * 0.05)).indices.tolist()

device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
        "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1",
    padding_side="left",
    truncation_side="right",
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def process(df, labels=True):
    lst = []
    for idx, row in df.iterrows():
        context = f"### Input: {row['context']}"

        text = "\n\n ### System: Answer the following multiple choice question by giving the most appropriate response. "
        text += "Answer should be one among [A, B, C, D, E]. Use the input text above as a reference for your answers if needed. "
        text += f"Question: {row['prompt']} \n "
        for choice in 'ABCDE':
            text += f'{choice}) {row[choice]} \n '
        text += "\n ### Answer:"

        lst.append({'context': context, 'text': text})
        
    return pd.DataFrame(lst, columns=['context', 'text'])

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

label_2_ids = {}
for x in "ABCDE":
    label_2_ids[x] = tokenizer(f"_{x}").input_ids[2]
label_2_ids

wrost_question = test_df_2.iloc[worst_question_ids]
test = process(wrost_question)
test_ds = Dataset.from_pandas(test)
test_dl = DataLoader(test_ds, batch_size=1)

pred_2 = None
for batch in tqdm(test_dl, total=len(test_dl)):
    tok = tokenizer(batch["context"], batch['text'], padding=False, truncation='only_first', return_tensors='pt', max_length=1024).to("cuda")
    output = model.generate(**tok, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    logit = output.scores[0].detach().cpu().numpy()
    logit = logit[:,[label_2_ids[choice] for choice in "ABCDE"]]
    if pred_2 is None:
        pred_2 = logit
    else:
        pred_2 = np.append(pred_2, logit, axis=0)


pred = pred_2
predictions_as_ids = np.argsort(-pred)[:,:3]
predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
predictions_as_string = [
    ' '.join(row) for row in predictions_as_answer_letters
]

tmp = pd.read_csv("prediction1.csv")
orgin_pred = tmp.prediction.copy()
orgin_pred[worst_question_ids] = predictions_as_string
tmp["prediction"] = orgin_pred
tmp.to_csv('submission.csv', index=False)

try:
    train_df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
    if len(train_df) == 200:
        m = MAP_at_3(sub.prediction.values, train_df.answer.values)
        print( 'CV MAP@3 =',m )
except Exception as e:
    print(e)