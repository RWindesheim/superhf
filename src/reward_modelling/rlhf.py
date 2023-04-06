from dataclasses import dataclass, field
from typing import Optional 

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from reward_model import RewardModel, PreferenceDataCollator
from preference_datasets import AnthropicHelpfulHarmless


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

  
    model_name: Optional[str] = field(default="lvwerra/gpt2-imdb", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=16, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
)



sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}



def build_dataset(dataset_name, data_dir=None, split='train'):
    dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)
    dataset = dataset.rename_columns({'chosen': 'query'})
    return dataset

class LMCollator:
    def __init__(self, tokenizer, padding=True, max_length=512):
        self.tokenizer = tokenizer
        self.padding=padding
        self.max_length=max_length

    def __call__(self, batch):
        prompts = []
        for item in batch:
            prompts.append(item['query'])
        prompts = [p.split("\n\nAssistant:")[0] + "\n\nAssistant:" for p in prompts]
        outputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        )
        output_dict = {}
        for k,v in outputs.items():
            output_dict[k] = [v_ for v_ in v]
        return output_dict
   
dataset = build_dataset("Anthropic/hh-rlhf", data_dir='harmless-base', split='train')

set_seed(config.seed)


model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')

tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, 
    data_collator=LMCollator(tokenizer))

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  

rm_path = '/juice2/scr2/fongsu/reward_model_HH/2023-04-04 23:27:29.175109/pytorch_model.bin'

reward_model_name_or_path = 'distilbert-base-uncased'
reward_model = RewardModel(reward_model_name_or_path)
reward_model.load_state_dict(torch.load(rm_path))
reward_model.to(device)
reward_model.eval()

rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_or_path, max_length=512)
if rm_tokenizer.pad_token == None:
    rm_tokenizer.pad_token = tokenizer.eos_token

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Adding the weighing
    prompts = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
    w1_list = [random.uniform(0, 1) for _ in range(len(prompts))]
    w2_list = [1 - w1 for w1 in w1_list]
    prompts = [f"{prompt} [W1={w1:.2f}][W2={w2:.2f}]" for prompt, w1, w2 in zip(prompts, w1_list, w2_list)]
    query_tensors = tokenizer.batch_encode_plus(prompts, padding=True, max_length=512, return_tensors="pt")["input_ids"]

    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)

    outputs1 = reward_model(**rm_tokenizer(batch['response'], return_tensors='pt', padding=True, truncation=True).to(device))
    rewards1 = [torch.tensor(output) for output in outputs1.tolist()]

    outputs2 = reward_model2(**rm_tokenizer2(batch['response'], return_tensors='pt', padding=True, truncation=True).to(device))
    rewards2 = [torch.tensor(output) for output in outputs2.tolist()]

    # Combine weighting + reward
    rewards = [w1 * r1 + w2 * r2 for w1, w2, r1, r2 in zip(w1_list, w2_list, rewards1, rewards2)]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
