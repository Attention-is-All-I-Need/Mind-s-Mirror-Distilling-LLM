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


import argparse
import re
import json
import numpy as np
import json
from datasets import Dataset, DatasetDict, load_dataset


DATASET_ROOT = 'datasets'

def filter_string(input_str):
    return re.sub(r'[^0-9\+\-\*\/\.\(\) ]', '', input_str)

class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        print("dataset_name: ",dataset_name)
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }
        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})
        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)
        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])
        return datasets

    
    def load_llm_preds_tree1(self, split):
        if self.dataset_name in ("anli1", "anli1_tree1", "anli1_tree2"):
            print("open path: ",f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT.json')
            # regular expression is used to extract answers
            pattern0 = re.compile(r'the answer is [\'"]?entailment[\'"]?', re.IGNORECASE)
            pattern1 = re.compile(r'the answer is [\'"]?neutral[\'"]?', re.IGNORECASE)
            pattern2 = re.compile(r'the answer is [\'"]?contradiction[\'"]?', re.IGNORECASE)
            labels = list()
            rationales = list()
            for idx in getattr(self, f'{split}_batch_idxs'):
                with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                    # print(f)
                    outputs = json.load(f)
                for output in outputs:
                    for y in output['steps']:
                        s = y['new_ys']
                        for idx in range(0,5): # Please adjust the range here according to the number of CoT/num_train_epochs
                            sentence = s[idx]
                            sentences_list = sentence.split("the answer is ")
                            new_sentence = "the answer is ".join(sentences_list[:-1])
                            last_period_index = new_sentence.rfind('.')
                            rationale = new_sentence[:last_period_index+1].strip()
                            if pattern0.search(s[idx]):
                                label = 'entailment'
                            elif pattern1.search(s[idx]):
                                label = 'neutral'
                            elif pattern2.search(s[idx]):
                                label = 'contradiction'
                            else:
                                label = ' '
                                rationale = ' '
                            rationales.append(rationale)
                            labels.append(label)
            
            return rationales, labels
        elif self.dataset_name in ("cqa", "cqa_tree1", "cqa_tree2"):
            print("open path: ",f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT.json')
            # regular expression is used to extract answers
            pattern = r'the answer is ("[^"]+"|[a-zA-Z ]+)'
            labels = list()
            rationales = list()
            for idx in getattr(self, f'{split}_batch_idxs'):
                with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                    outputs = json.load(f)
                for output in outputs:
                    for y in output['steps']:
                        s = y['new_ys']
                        for idx in range(0,5): # Please adjust the range here according to the number of CoT/num_train_epochs
                            sentence = s[idx]
                            sentences_list = sentence.split("the answer is ")
                            new_sentence = "the answer is ".join(sentences_list[:-1])
                            last_period_index = new_sentence.rfind('.')
                            rationale = new_sentence[:last_period_index+1].strip()
                            match = re.search(pattern, s[idx])
                            if match:
                                label = match.group(1).strip('"').strip()
                            else:
                                label = ' '
                                rationale = ' '
                            rationales.append(rationale)
                            labels.append(label)
            return rationales, labels
        elif self.dataset_name in ("svamp", "svamp_tree1", "svamp_tree2"):
            print("open path: ",f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT.json')
            pattern = r"answer is (.*?)\."
            labels = list()
            rationales = list()
            for idx in getattr(self, f'{split}_batch_idxs'):
                with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                    outputs = json.load(f)
                for output in outputs:
                    for y in output['steps']:
                        s = y['new_ys']
                        for idx in range(0,5): # Please adjust the range here according to the number of CoT/num_train_epochs
                            sentence = s[idx]
                            sentences_list = sentence.split("answer is ")
                            new_sentence = "answer is ".join(sentences_list[:-1])
                            last_period_index = new_sentence.rfind('.')
                            rationale = new_sentence[:last_period_index+1].strip()

                            match = re.search(pattern, s[idx])
                            if match:
                                label = match.group(1)
                                label = label.split('=')[0].strip()
                                label = filter_string(label)
                            else:
                                label = ' '
                                rationale = ' '
                            print("rationale:",rationale, sep='')
                            print("label:",label, sep='')
                            rationales.append(rationale)
                            labels.append(label)
            return rationales, labels
        
    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)
            for output in outputs:
                rationale, label = self._parse_llm_output(output)
                rationales.append(rationale)
                labels.append(label)
        return rationales, labels

    def load_llm_preds_tree2(self, split): # Extraction of labels and rationales for self-evaluation
        print("open path: ",f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT.json')
        # regular expression is used to extract answers
        pattern0 = re.compile(r'given answer is [\'"]?uncertain[\'"]?', re.IGNORECASE)
        pattern1 = re.compile(r'given answer is [\'"]?wrong[\'"]?', re.IGNORECASE)
        pattern2 = re.compile(r'given answer is [\'"]?correct[\'"]?', re.IGNORECASE)
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)
            for output in outputs:
                for y in output['steps']:
                    for s_index in range(5): # Please adjust the range here according to the number of CoT/num_train_epochs, and replace the xx_tree2_xx.json file with the corresponding number
                        s = y['value_outputs_list'][s_index]
                        for idx in range(0,5): # Please adjust the range here according to the number of self-evaluations
                            sentence = s[idx]
                            sentences_list = sentence.split("\n")
                            new_sentence = "\n".join(sentences_list[:-1])
                            rationale = new_sentence.strip()
                            if pattern0.search(s[idx]):
                                label = 'uncertain'
                            elif pattern1.search(s[idx]):
                                label = 'wrong'
                            elif pattern2.search(s[idx]):
                                label = 'correct'
                            else:
                                continue
                            rationales.append(rationale)
                            labels.append(label)
        return rationales, labels
    
    
    
    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class CQADatasetLoader(DatasetLoader):
    def __init__(self, dataset_name):
        source_dataset_name = 'cos_e'
        dataset_version = 'v1.11'
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'validation',
        }
        # The values here need to be adjusted according to the amount of data and the number of JSON files in the llm folder.
        batch_size = 210000
        train_batch_idxs = range(4)
        test_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]
            c_4 = example['choices'][4]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['id', 'question', 'choices', 'answer', 'abstractive_explanation', 'extractive_explanation'])

        return datasets



class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name):
        #dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        # The values here need to be adjusted according to the amount of data and the number of JSON files in the llm folder.
        batch_size = 50000
        train_batch_idxs = range(1)
        test_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/SVAMP.json') as f:
            original_dataset = json.load(f)

        dataset = list()
        for data in original_dataset:
            input = f'{data["Body"]}\n{data["Question"]}'
            equation = data["Equation"]

            dataset.append({
                'input': input,
                'label': equation,
            })

        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        train_idxs = idxs[:800]
        test_idxs = idxs[800:]

        train_dataset = Dataset.from_list(np.array(dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets
        

    def _post_process(self, datasets):
        return datasets

class ANLIDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs):

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):
        
        def label_idx2text(example):            
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'
            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(['uid', 'reason'])

        return datasets


class ANLI1DatasetLoader(ANLIDatasetLoader):
    def __init__(self, dataset_name):
        #dataset_name = 'anli1'
        source_dataset_name = 'anli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train_r1',
            'valid': 'dev_r1',
            'test': 'test_r1',
        }
        # The values here need to be adjusted according to the amount of data and the number of JSON files in the llm folder.
        batch_size = 210000
        train_batch_idxs = range(7)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'cqa' or args.dataset == 'cqa_tree1' or args.dataset == 'cqa_tree2':
        dataset_loader = CQADatasetLoader(dataset_name=args.dataset)
    elif args.dataset == 'svamp' or args.dataset == 'svamp_tree1' or args.dataset == 'svamp_tree2':
        dataset_loader = SVAMPDatasetLoader(dataset_name=args.dataset)
    elif args.dataset == 'anli1' or args.dataset == 'anli1_tree1' or args.dataset == 'anli1_tree2':
        dataset_loader = ANLI1DatasetLoader(dataset_name=args.dataset)

    datasets = dataset_loader.load_from_source()
    dataset_loader.to_json(datasets)
