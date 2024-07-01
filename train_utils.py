# Copyright 2024 The Mind's Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This project partially references the project (https://github.com/google-research/distilling-step-by-step) by the Distilling-step-by-step authors


import os
import shutil
import logging

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed

from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer


def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}_continue_train_{str(args.continue_train)}/{args.model_type}_{args.llm}_num_train_branches_{args.num_train_branches}/subsample_{args.subsample}/label_type_{args.label_type}_prompt_{str(args.prompt)}/alpha_{args.alpha}_output_rationale_{str(args.output_rationale)}/{args.max_input_length}_{args.grad_steps*args.batch_size}_{args.optimizer_name}_{args.lr}/num_train_epochs_{args.num_train_epochs}'


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    print("args.prompt: ", args.prompt)
    print("type args.prompt: ", type(args.prompt))
    print("args.from_pretrained: ", args.from_pretrained)
    print("type args.from_pretrained: ", type(args.from_pretrained))

    if args.continue_train == True:
        print("-------------Continue training after completing self-evaluation capability training-------------")
        trained_model_dir = "/" # Replace with the path of the best model trained with self-evaluation capability
        print("trained_model_dir: ", trained_model_dir)
        model = T5ForConditionalGeneration.from_pretrained(trained_model_dir)
    else:
        print("-------------Training a new model-------------")
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
    
    if args.parallelize:
        model.parallelize()
    
    config_dir = get_config_dir(args)
    output_dir = f'ckpts/{config_dir}/seed_{run}'
    logging_dir = f'logs/{config_dir}/seed_{run}'
    
    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.error('Found existing ckpt directory. Exiting to avoid overwriting it.')
        raise Exception('Directory already exists: ' + output_dir)

    
    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = 'epoch',
        save_strategy='epoch',
        save_total_limit=3,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=250,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=args.local_rank,
        bf16=args.bf16,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_test_accuracy",
        greater_is_better=True,
    )
    if args.model_type == 'task_prefix' or args.model_type == 'task_prefix_tree1' or args.model_type == 'task_prefix_tree2':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError


    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    if args.model_type == 'task_prefix' or args.model_type == 'task_prefix_tree1' or args.model_type == 'task_prefix_tree2':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError
    

    trainer.train()
    trainer.save_model(f'best_model/{config_dir}/seed_{run}')
    