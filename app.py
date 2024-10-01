from datasets import load_dataset,Dataset
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
dataset = load_dataset(dataset_name, split='train')
device = torch.device("cpu")
df_train = dataset.to_pandas()
train_data, eval_data = train_test_split(df_train, test_size=0.2, random_state=42)
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def preprocess_function(sample, padding="max_length"):
    model_inputs = tokenizer(sample["instruction"], max_length=256, padding=padding, truncation=True)
    labels = tokenizer(sample["response"], max_length=256, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized_dataset = Dataset.from_pandas(train_data).map(preprocess_function, batched=True, remove_columns=['flags', 'instruction', 'category', 'intent', 'response'])
test_tokenized_dataset = Dataset.from_pandas(eval_data).map(preprocess_function, batched=True, remove_columns=['flags', 'instruction', 'category', 'intent', 'response'])
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
)
output_dir = "lora-flan-t5-small-chat"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    push_to_hub=False
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
)
model.config.use_cache = False
trainer.train()
original_tokenizer = AutoTokenizer.from_pretrained(model_id)

sample = "Human: \n How can I cancel my order?"
input_ids = original_tokenizer(sample, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)

peft_model_id = "[path_to_your_project]/checkpoint-13440"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0}).to(device)
model.eval()

outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_length=256)

print("output:")
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

