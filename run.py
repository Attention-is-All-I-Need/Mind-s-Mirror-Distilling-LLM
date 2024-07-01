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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import argparse

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ANLI1DatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate

import pandas as pd
def run(args):
    # Prepare datasets
    if args.dataset == 'cqa' or args.dataset == 'cqa_tree1' or args.dataset == 'cqa_tree2':
        dataset_loader = CQADatasetLoader(dataset_name=args.dataset)
    elif args.dataset == 'svamp' or args.dataset == 'svamp_tree1' or args.dataset == 'svamp_tree2':
        dataset_loader = SVAMPDatasetLoader(dataset_name=args.dataset)
    elif args.dataset == 'anli1' or args.dataset == 'anli1_tree1' or args.dataset == 'anli1_tree2':
        dataset_loader = ANLI1DatasetLoader(dataset_name=args.dataset)
    else:
        raise ValueError

    datasets = dataset_loader.load_from_json()

    if args.model_type == 'task_prefix_tree1':
        for key, data in datasets.items():
            expanded_data = []
            for instance in data:
                for _ in range(args.num_train_branches):
                    expanded_data.append(instance)
            expanded_df = pd.DataFrame(expanded_data)
            datasets[key] = Dataset.from_pandas(expanded_df)
    
    if args.llm is None:
        pass
    elif args.llm == 'gpt-3.5-turbo':
        if args.model_type == 'task_prefix_tree1':
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds_tree1(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds_tree1(split='test')
        elif args.model_type == 'task_prefix_tree2':
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds_tree2(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds_tree2(split='test')
        elif args.model_type == 'standard':
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds_tree1(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds_tree1(split='test')
    elif args.llm == 'gpt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
    else:
        raise ValueError
    
    if args.llm is not None:
        datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
        datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)
        datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
        datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)
        
    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']
        
    if dataset_loader.has_valid:
        if args.llm is None:
            pass
        elif args.llm == 'gpt-3.5-turbo':
            if args.model_type == 'task_prefix_tree1':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds_tree1(split='valid')
            elif args.model_type == 'task_prefix_tree2':
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds_tree2(split='valid')
            else:
                valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
        elif args.llm == 'gpt':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
        else:
            raise ValueError

        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
    else:
        train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': train_valid_datasets['train'],
            'valid': train_valid_datasets['test'],
            'test': datasets['test'],
        })
    if args.label_type == 'gt':
        pass
    elif args.label_type == 'llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'svamp_tree1', 'svamp_tree2'] or args.model_type == "task_prefix_tree2":
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])
    else:
        raise ValueError
    if args.llm is not None:
        if 'rationale' in datasets['train'].column_names:
            datasets = datasets.remove_columns('rationale')
        datasets = datasets.rename_column('llm_rationale', 'rationale')

    print("args.prompt: ", args.prompt)
    print("type args.prompt: ", type(args.prompt))

    # Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )
    if args.model_type == 'task_prefix_tree1' and args.llm is not None:
        def tokenize_function(examples):
            
            if args.prompt == True:
                print("-----------------prompt is used-----------------")
                if 'nli' in args.dataset:
                    prompt = ''' '''
                elif args.dataset == 'cqa' or args.dataset == 'cqa_tree1':
                    prompt = ''' '''
                elif args.dataset == 'svamp' or args.dataset == 'svamp_tree1':
                    prompt = ''' '''
            else:
                prompt = ""

            print("prompt:", prompt)
            model_inputs = tokenizer([prompt + 'predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer([prompt + 'explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
                
                
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            return model_inputs
    elif args.model_type == 'task_prefix_tree2' and args.llm is not None:
        def tokenize_function(examples):
            if args.prompt == True:
                print("-----------------prompt is used-----------------")
                if 'nli' in args.dataset:
                    prompt = ''' '''
                elif args.dataset == 'cqa' or args.dataset == 'cqa_tree1':
                    prompt = ''' '''
                elif args.dataset == 'svamp' or args.dataset == 'svamp_tree1':
                    prompt = ''' '''
            else:
                prompt = ""
            
            print("prompt:", prompt)

            model_inputs = tokenizer([prompt + 'predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer([prompt + 'explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
                
                
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            return model_inputs
    elif args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            if args.prompt == True:
                print("-----------------prompt is used-----------------")
                if 'nli' in args.dataset:
                    prompt = ''' '''
                elif args.dataset == 'cqa' or args.dataset == 'cqa_tree1':
                    prompt = ''' '''
                elif args.dataset == 'svamp' or args.dataset == 'svamp_tree1':
                    prompt = ''' '''
            else:
                prompt = ""
            
            print("prompt:", prompt)

            model_inputs = tokenizer([prompt + 'predict: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer([prompt + 'explain: ' + text for text in examples['input']], max_length=args.max_input_length, truncation=True)
            
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
            
            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['rationale'], max_length=256, truncation=True)
                
                
            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']
            return model_inputs
    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError
    
    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'rationale', 'label', 'llm_label'],
            batched=True
        )


    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'svamp_tree1']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset not in ['svamp', 'svamp_tree1']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)
    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)
    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=256)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--num_train_branches', type=int, default=5, help='Number of training branches')
    parser.add_argument('--prompt', action='store_true', help='Control whether to use the prompt or not')
    parser.add_argument('--continue_train', action='store_true', help='Control whether to use an already trained model to continue training')
    args = parser.parse_args()

    run(args)