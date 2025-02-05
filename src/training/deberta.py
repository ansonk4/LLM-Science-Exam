"""
This script fine-tunes a DeBERTa model for a multiple-choice question answering task using the Hugging Face Transformers library and PEFT (Parameter-Efficient Fine-Tuning). The script includes data preprocessing, model loading, training, and evaluation with the following key components:
1. **Preprocessing**: Tokenizes input data and prepares it for the model.
2. **Data Collator**: Custom data collator for handling multiple-choice inputs.
3. **Metrics**: Computes MAP@3 score for model evaluation.
4. **Model Loading**: Loads a pre-trained DeBERTa model and applies LoRA (Low-Rank Adaptation) for efficient fine-tuning.
5. **Callbacks**: Custom callback for saving the PEFT model during training.
6. **Training**: Sets up and runs the training loop with specified hyperparameters and evaluation strategy.
Functions and Classes:
- `preprocess(example)`: Preprocesses a single example for the multiple-choice task.
- `DataCollatorForMultipleChoice`: Custom data collator for padding and batching multiple-choice inputs.
- `map_at_3(predictions, labels)`: Computes the MAP@3 score for evaluation.
- `compute_metrics(p)`: Wrapper function to compute metrics during evaluation.
- `load_data(train_path, valid_path)`: Loads and preprocesses training and validation datasets.
- `load_model(model_name)`: Loads and configures the DeBERTa model with LoRA.
- `SavePeftModelCallback`: Custom callback for saving the PEFT model during training.
- `main()`: Main function to execute the training pipeline.
Usage:
Run the script as a standalone program to start training the model.
"""
 
from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer, AutoModelForMultipleChoice
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
from kaggle_secrets import UserSecretsClient
import wandb

user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("wandb") 
wandb.login(key=wandb_api)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", padding_side="right", truncation_side="right")

use_wandb = True

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [ "[CLS] " + example['context'] ] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] if example[option] is not None else "Nan" + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first', padding='max_length',
                                  max_length=512, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
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
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

# function for compting the score MAP@3 during validion
def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}

def load_data(train_path, valid_path):
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path, index_col=[0])[:200]
    valid_ds = Dataset.from_pandas(df_valid)
    train_ds = Dataset.from_pandas(df_train)
    tok_valid_ds = valid_ds.map(preprocess, remove_columns=list(valid_ds.features.keys()))
    tok_train_ds = train_ds.map(preprocess, remove_columns=list(train_ds.features.keys()))

    return tok_valid_ds, tok_train_ds

def load_model(model_name):
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["query_proj", "value_proj"],
        task_type=TaskType.SEQ_CLS,
        inference_mode=False, 
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier", "pooler"],
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    return model

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

def main():
    train = '/kaggle/input/60k-data-with-context-v2/all_12_with_context2.csv'
    valid = '/kaggle/input/faiss-k5-l20/valid.csv'
    tok_valid_ds, tok_train_ds = load_data()
    model = load_model("microsoft/deberta-v3-large")
    
    training_args = TrainingArguments(
        warmup_ratio=0.1, 
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        output_dir = f'./checkpoints',
        overwrite_output_dir=True,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=50,
        save_strategy="steps",
        save_steps=25,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        save_total_limit=2,
        report_to= "wandb" if use_wandb else "none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tok_train_ds,
        eval_dataset=tok_valid_ds,
        compute_metrics=compute_metrics,
        callbacks=[SavePeftModelCallback],
    )
    
    trainer.train(resume_from_checkpoint=None)
    
    model.save_pretrained("model")


if __name__ == "__main__":
    main()