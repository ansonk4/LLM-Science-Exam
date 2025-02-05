import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import torch.nn.functional as F
import os
from kaggle_secrets import UserSecretsClient
import wandb

user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("wandb") 
wandb.login(key=wandb_api)
use_wandb = True

train = pd.read_csv("/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv")
valid = pd.read_csv("/kaggle/input/faiss-k5-l20/valid.csv", index_col=[0])[:200]

def process(df, labels=True):
    lst = []
    for idx, row in df.iterrows():
            text = f"{row['context'][:2500]}\n"
            text += f"Question: {row['prompt']}\n\n"
            text += f"A. {row['A']}\n"
            text += f"B. {row['B']}\n"
            text += f"C. {row['C']}\n"
            text += f"D. {row['D']}\n"
            text += f"E. {row['E']}\n"
            text += f" ###Answer: {row['answer']}"
            lst.append({'text': text})
        
    return pd.DataFrame(lst, columns=['text'])


train, valid = process(train), process(valid)
train_ds = Dataset.from_pandas(train)
valid_ds = Dataset.from_pandas(valid)
train_ds

tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1",
    padding_side="right",
    truncation_side="left",
)

def tok(row):
    ret = tokenizer(row["text"], padding=False, truncation=False)
    return ret

train_tok = train_ds.map(tok, batched=True)
train_ds = train_tok.filter(lambda example: len(example['input_ids']) <= 1024)

tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1",
    model_max_length=1024,
    padding_side="right",
    truncation_side="left",
)

tokenizer.pad_token = tokenizer.unk_token

bnb_config = BitsAndBytesConfig(  
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant= True,
)

model = AutoModelForCausalLM.from_pretrained(
        "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1",
        num_labels=1,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
)

model.config.pad_token_id = tokenizer(tokenizer.unk_token)["input_ids"][1]

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8, lora_alpha=8, task_type=TaskType.CAUSAL_LM, lora_dropout=0.1, 
    bias="none", inference_mode=False, 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):   
    ids_to_choices = {330: 0, 365: 1, 334: 2, 384: 3, 413: 4}
    
    logits, raw_labels = p
    labels = [] #(eval_size) [B, A, ... , E, A]
    for label in raw_labels:
        if label[-1] in ids_to_choices.keys():
            labels.append(ids_to_choices[label[-1]])
        else: # when padding token exist at the end of the sentence, loop from the end to found the label
            found = False
            for x in reversed(label):
                if x in ids_to_choices.keys():
                    labels.append(ids_to_choices[x])
                    found = True
                    break
            if not found:
                labels.append(ids_to_choices[330])
    
    logits = [ [logit[ids] for ids in ids_to_choices.keys()] for logit in logits ] #(eval_size, 5)
    return {"map@3": map_at_3(logits, labels)}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[:,-1,:]

# call back for saving adapters
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

training_args = TrainingArguments(
    warmup_ratio=0.1, 
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    output_dir = f'./checkpoints',
    overwrite_output_dir=True,
    optim="paged_adamw_32bit",
    fp16=True,
    gradient_accumulation_steps=1,
    logging_steps=25,
    evaluation_strategy='steps',
    eval_steps=200,
    save_strategy="steps",
    save_steps=25,
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    save_total_limit=2,
    report_to= "wandb" if use_wandb else "none",  # enable logging to W&B
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    dataset_text_field="text",
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                  response_template="###Answer:"),
    callbacks=[SavePeftModelCallback],
)

trainer.train(resume_from_checkpoint="/kaggle/input/mistral-5-label-checkpoint/checkpoints/checkpoint-5725")

model.save_pretrained("model")