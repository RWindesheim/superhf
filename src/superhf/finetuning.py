"""
Trainers for finetuning models according to SuperHF (maximizing the scores
from a reward model without the use of reinforcement learning).
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Any, Optional, Tuple

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
    EvalPrediction,
)  # type: ignore
from transformers.pipelines.pt_utils import KeyDataset
from accelerate import Accelerator
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from datasets.arrow_dataset import Dataset
from datasets.utils import logging

# from torchtyping import TensorType
logger = logging.get_logger(__name__)


def print_gpu_utilization() -> None:
    """
    Print the GPU memory occupied using nvidia-smi.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


@dataclass
class SuperHFTrainingArguments(TrainingArguments):
    """
    Training arguments for SuperHF trainers.
    """

    reward_model: Optional[nn.Module] = None


# pylint: disable=too-few-public-methods
class SuperHFTrainer(ABC):
    """
    Base class for SuperHF trainers.

    Fine-tuned a language model to maximize the scores from a reward model.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        language_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        language_tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: PreTrainedTokenizerBase,
        train_prompts: List[str],
        test_prompts: List[str],
    ) -> None:
        self.language_model = language_model
        self.reward_model = reward_model
        self.language_tokenizer = language_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.train_prompts = train_prompts
        self.test_prompts = test_prompts

    @abstractmethod
    def train(self) -> None:
        """
        The main training and evaluation loop.
        """
        raise NotImplementedError


# pylint: disable=too-many-instance-attributes
class SinglePassBestOfNTrainer(SuperHFTrainer):
    """
    The most basic form of Super HF: filtering completions by the reward model
    and fine-tuning the language model on the filtered completions.

    As one long training sequence
        1. Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts
              ($d*completions_per_prompt$ total).
        2. Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        3. Fine-tune $M$ on the $d$ best-of-$n$ completions.
        4. Evaluate the loss and average reward during training.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        models: Dict[str, PreTrainedModel],
        tokenizers: Dict[str, PreTrainedTokenizerBase],
        data: Dict[str, List[str]],
        temperature: float = 0.7,
        completions_per_prompt: int = 4,
        output_dir: str = "output",
        debug: bool = False,
    ) -> None:
        super().__init__(
            models["language_model"],
            models["reward_model"],
            tokenizers["language_tokenizer"],
            tokenizers["reward_tokenizer"],
            data["train_prompts"],
            data["test_prompts"],
        )
        self.temperature = temperature
        self.completions_per_prompt = completions_per_prompt
        self.output_dir = output_dir
        self.training_args: Any = None
        self.eval_dataset: Any = None
        self.debug = debug
        # Make output dir if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self) -> None:
        """
        Main training and evaluation loop.
        """

    def generate_completions(self, batch_size: int, max_new_tokens: int) -> None:
        """
        Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts.
        """
        # Set up tokenizer
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        # Switch to eval mode
        self.language_model.eval()

        if self.debug:
            self.train_prompts = self.train_prompts[:124]

        # Duplicate each of the prompts $n$ times.
        prompts = self.train_prompts * self.completions_per_prompt

        # Convert prompts into a train_dataset
        train_dataset = Dataset.from_dict({"prompt": prompts})

        # Use $M$ to generate $n$ completions for each of $d$ training train_dataset prompts
        # ($d*completions_per_prompt$ total). Iterate in groups of batch_size.
        pipe = pipeline(
            "text-generation",
            model=self.language_model,
            tokenizer=self.language_tokenizer,
            device=self.language_model.device,
        )

        logger.info("Generating completions...")
        print("Generating completions...")
        completions: List[str] = []

        for out in tqdm(
            pipe(
                KeyDataset(train_dataset, "prompt"),
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                pad_token_id=self.language_tokenizer.pad_token_id,
                early_stopping=True,
                do_sample=True,
            ),
            total=len(train_dataset),
        ):
            completion = out[0]["generated_text"]
            # Filter out everything including and after the second "\n\nHuman:"
            # sequence in case the model repeats the prompt
            completion = "\n\nHuman:".join(completion.split("\n\nHuman:")[:2])
            completion = "\n\nAssistant:".join(completion.split("\n\nAssistant:")[:2])
            completions.append(completion)

        # Save it to a file, writing raw string outputs (e.g. keeping '\n' in plaintext)
        torch.save(completions, os.path.join(self.output_dir, "completions.pt"))

    def score_completions(self, batch_size: int) -> None:
        """
        Use $R$ to evaluate each completion.
        """

        completions: List[str] = torch.load(
            os.path.join(self.output_dir, "completions.pt")
        )

        num_prompts: int = len(self.train_prompts)
        # Debug: only use a subset of the completions
        # if self.debug:
        #     completions = [
        #         completion
        #         for i, completion in enumerate(completions)
        #         if i % num_prompts < 1024
        #     ]

        # OOM Fix: Filter completions in a set longer than 1000 characters
        bad_indices = []
        for i, completion in enumerate(completions):
            if len(completion) > 1000:
                bad_indices.append(i % num_prompts)
        old_size = len(completions)
        completions = [
            completion
            for i, completion in enumerate(completions)
            if i % num_prompts not in bad_indices
        ]
        new_size = len(completions)
        print(
            (
                f"Loaded {int(new_size / old_size * 100)}% ({new_size} completions)"
                f" (filtered {int(100 - new_size / old_size * 100)}% from"
                f" {old_size} total)"
            ),
        )

        train_dataset = Dataset.from_dict({"completion": completions})

        # Use $R$ to select the top 1 of the $n$ completions for each prompt ($d$ total).
        pipe = pipeline(
            "text-classification",
            model=self.reward_model,
            tokenizer=self.reward_tokenizer,
            device=self.reward_model.device,
        )
        scored_completions: List[Dict[str, Union[str, float]]] = []
        print(f"There are {len(train_dataset)} completions to score.")
        print("Scoring completions...")
        for row, completion in zip(
            tqdm(
                pipe(
                    KeyDataset(train_dataset, "completion"),
                    batch_size=batch_size,
                    max_length=512,
                ),
                total=len(train_dataset),
            ),
            completions,
        ):
            scored_completions.append({"score": row["score"], "completion": completion})

        print(
            "Average reward:"
            f" {np.mean([float(row['score']) for row in scored_completions])}"
        )

        torch.save(
            scored_completions,
            os.path.join(self.output_dir, "scored_completions.pt"),
        )

    def filter_completions(
        self,
    ) -> Tuple[Any, List[Dict[str, Union[str, float]]]]:
        """
        Select the top 1 of the $n$ completions for each prompt ($d$ total).
        """
        scored_completions: List[Dict[str, Union[str, float]]] = torch.load(
            os.path.join(self.output_dir, "scored_completions.pt")
        )
        num_prompts: int = len(scored_completions) // self.completions_per_prompt

        # Group the completions for the same prompt together into a list of lists.
        # E.g. ['a1', 'b1', 'c1', 'a2', 'b2', 'c2'] -> [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
        grouped_scored_completions: List[List[Dict[str, Union[str, float]]]] = [
            [] for _ in range(num_prompts)
        ]
        for i, scored_completion in enumerate(scored_completions):
            grouped_scored_completions[i % num_prompts].append(scored_completion)

        # Filter for the best completion for each group
        filtered_completions: List[Dict[str, Union[str, float]]] = []
        for group in grouped_scored_completions:
            filtered_completions.append(max(group, key=lambda x: x["score"]))

        print(
            f"Saving {len(filtered_completions)} filtered completions to"
            f" {os.path.join(self.output_dir, 'filtered_completions.pt')}"
        )
        torch.save(
            filtered_completions,
            os.path.join(self.output_dir, "filtered_completions.pt"),
        )
        return scored_completions, filtered_completions

    def load_filtered_completions(
        self,
    ) -> Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, Union[str, float]]]]:
        """
        Read the filtered completions and scored completions from disk.
        """
        filtered_completions: List[Dict[str, Union[str, float]]] = torch.load(
            os.path.join(self.output_dir, "filtered_completions.pt")
        )
        scored_completions: List[Dict[str, Union[str, float]]] = torch.load(
            os.path.join(self.output_dir, "scored_completions.pt")
        )
        return scored_completions, filtered_completions

    def tune_model(self, training_args: TrainingArguments) -> None:
        """
        Fine-tune $M$ on the $d$ best-of-$n$ completions.
        Evaluate the loss and average reward during training.
        """
        self.training_args = training_args
        assert (
            self.output_dir is not None
        ), "Must specify output_dir both for loading completions and saving the model"
        assert self.test_prompts is not None, "Must specify test_prompts"
        print_gpu_utilization()
        filtered_completions: List[Dict[str, Union[str, float]]] = torch.load(
            os.path.join(self.output_dir, "filtered_completions.pt")
        )
        print(
            f"Loaded {len(filtered_completions)} filtered completions from "
            f"{os.path.join(self.output_dir, 'filtered_completions.pt')}"
        )
        print_gpu_utilization()
        if self.debug:
            filtered_completions = filtered_completions[:1024]
            print(
                "Debug: only using a subset of the completions, len=",
                len(filtered_completions),
            )
        print(type(filtered_completions))
        print(type(self.test_prompts))
        # train_dataset = Dataset.from_dict({"completion": filtered_completions})
        # eval_dataset = Dataset.from_dict({"prompt": self.test_prompts})

        train_dataloader = DataLoader(
            filtered_completions, batch_size=training_args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            self.test_prompts, batch_size=training_args.per_device_eval_batch_size
        )
        self.eval_dataset = eval_dataloader

        # print("Pre-processing datasets...")
        print(type(train_dataloader))
        logging.enable_progress_bar()
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        # train_dataset_processed = []
        # for i in tqdm(range(len(train_dataloader))):
        #     train_dataset_processed.append(
        #         self.language_tokenizer(
        #             train_dataloader[i]["completion"]["completion"],
        #             truncation=True,
        #         )
        #     )
        print_gpu_utilization()
        # train_dataset_processed = train_dataset.map(
        #     lambda examples: self.language_tokenizer(
        #         [example["completion"] for example in examples["completion"]],
        #         truncation=True,
        #     ),
        #     batched=True,
        # )
        # print(
        #     "We passed the train dataset, it has a length of: ",
        #     len(train_dataset_processed),
        # )
        # test_dataset_processed = []
        # for i in tqdm(range(len(eval_dataloader))):
        #     test_dataset_processed.append(
        #         self.language_tokenizer(
        #             eval_dataloader[i]["prompt"],
        #             truncation=True,
        #         )
        #     )
        # test_dataset_processed = eval_dataset.map(
        #     lambda examples: self.language_tokenizer(
        #         list(examples["prompt"]),
        #         truncation=True,
        #     ),
        #     batched=True,
        # )
        # print(
        #     "We passed the test dataset, it has a length of: ",
        #     len(test_dataset_processed),
        # )
        print_gpu_utilization()
        print("Setting up training...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.language_tokenizer, mlm=False
        )
        self.compute_metrics(EvalPrediction(predictions=[], label_ids=[]))
        self.language_model.train()
        print("Starting training...")

        trainer = Trainer(
            model=self.language_model,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=eval_dataloader,
            compute_metrics=self.compute_metrics,
        )

        if training_args.gradient_checkpointing:
            trainer.model.gradient_checkpointing_enable()

        accelerator = Accelerator(mixed_precision=training_args.fp16)
        model, optimizer, train_dataloader = accelerator.prepare(
            trainer.model, torch.optim.AdamW, train_dataloader
        )

        model.train()
        for step, batch in enumerate(train_dataloader, start=1):
            loss = model(**batch).loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        eval_results = trainer.evaluate()
        print(eval_results)

    def compute_metrics(self, _: EvalPrediction) -> Dict[str, float]:
        """
        Compute the average reward of new completions on the test prompts.

        We ignore the predictions and labels because we need to generate full
        completions for the test prompts, which is not possible with the
        Trainer API.
        """
        print("Computing metrics...")
        pipe = pipeline(
            "text-generation",
            model=self.language_model,
            tokenizer=self.language_tokenizer,
            device=self.language_model.device,
        )

        print(
            "Generating completions on device in order to evaluate them: ",
            self.language_model.device,
        )
        completions: List[str] = []
        for out in tqdm(
            pipe(
                KeyDataset(self.eval_dataset, "prompt"),
                batch_size=self.training_args.eval_batch_size,
                max_new_tokens=256,
                # temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.language_tokenizer.pad_token_id,
                early_stopping=True,
            )
        ):
            completion = out[0]["generated_text"]
            # Filter out everything including and after the second "\n\nHuman:"
            # sequence in case the model repeats the prompt
            completion = "\n\nHuman:".join(completion.split("\n\nHuman:")[:2])
            completion = "\n\nAssistant:".join(completion.split("\n\nAssistant:")[:2])
            completions.append(completion)

        # OOM Fix: Filter completions in a set longer than 1000 characters
        previous_size = len(completions)
        completions = [
            completion for completion in completions if len(completion) < 1000
        ]
        new_size = len(completions)
        if new_size < previous_size:
            print(
                "Filtered %d completions (%.2f%% of %d total) to prevent OOM.",
                previous_size - new_size,
                100 * (previous_size - new_size) / previous_size,
                previous_size,
            )

        # Now evaluate the completions with the reward model
        completions_dataset = Dataset.from_dict({"completion": completions})

        pipe = pipeline(
            "text-classification",
            model=self.reward_model,
            tokenizer=self.reward_tokenizer,
            device=self.reward_model.device,
        )

        print(
            "Evaluating the completions on the reward model on device: ",
            self.reward_model.device,
        )
        scores: List[float] = []
        for row, completion in tqdm(
            zip(
                pipe(
                    KeyDataset(completions_dataset, "completion"),
                    batch_size=self.training_args.eval_batch_size,
                    max_length=512,
                ),
                completions,
            )
        ):
            scores.append(row["score"])

        # Print an example completion
        print(f"Example completion: {completions[0]}")
        print(f"Example score: {scores[0]}")

        average_reward = float(np.mean(scores))
        average_completion_length = float(
            np.mean([len(completion) for completion in completions])
        )
        return {
            "average_reward": average_reward,
            "average_completion_length": average_completion_length,
        }


# class IterativeBestOfNTrainer(SuperHFTrainer):
#     """
#     The most basic form of Super HF: filtering completions by the reward model
#     and fine-tuning the language model on the filtered completions.

#     Iteratively, in a loop, we:
#         1. Sample $p$ prompts from the training set without replacement.
#         2. Use $M$ to generate $n$ completions for each prompt ($p*completions_per_prompt$ total).
#         3. Use $R$ to select the top 1 of the $n$ completions for each prompt ($p$ total).
#         4. Fine-tune $M$ on the $p$ best-of-$n$ completions.
#         5. Store the fine-tuning loss and average reward across the $p$ best-of-$n$ completions.
#     """

#     def __init__(
#         self,
#         language_model: GenerationMixin,
#         reward_model: nn.Module,
#         language_tokenizer: Any,
#         reward_tokenizer: Any,
#         train_prompts: List[str],
#         test_prompts: List[str],
#         temperature: float = 1.0,
#         completions_per_prompt: int = 2,
#     ) -> None:
#         super().__init__(
#             language_model,
#             reward_model,
#             language_tokenizer,
#             reward_tokenizer,
#             train_prompts,
#             test_prompts,
#         )
#         self.temperature = temperature
#         self.completions_per_prompt = completions_per_prompt

#     def train(self) -> None:
#         """
#         Main training and evaluation loop.
#         """
#         raise NotImplementedError
