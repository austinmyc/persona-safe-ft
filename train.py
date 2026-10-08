import argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from dataset_json_utils import load_train_records

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    help="HuggingFace model id or local path")
parser.add_argument("--data_path", type=str, required=True,
                    help="Training data: JSON array (.json) or JSONL (.jsonl) with key 'messages'")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory for saved checkpoints")
parser.add_argument("--chat_template", type=str, default="chatml",
                    choices=["chatml", "mistral", "auto"],
                    help="chatml (Llama/Qwen), mistral, or auto (SmolLM native tokenizer)")
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=3407,
                    help="3407 = the seed used for every run in the submitted paper")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

if args.chat_template != "auto":
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=args.seed,
)

data = load_train_records(args.data_path)
dataset = Dataset.from_list(data)

def format_examples(examples):
    return {"text": [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in examples["messages"]
    ]}

dataset = dataset.map(format_examples, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=args.num_epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        seed=args.seed,
        output_dir=args.output_dir,
        save_strategy="epoch",
        save_total_limit=10,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    ),
)

trainer.train()
