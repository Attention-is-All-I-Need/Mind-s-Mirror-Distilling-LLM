# Mind-s-Mirror-Distilling-LLM
Code for [Mind's Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models](https://aclanthology.org/2024.naacl-long.376), NAACL 2024

## Environment Setup
- Please follow these steps to create a virtual environment, replacing `your_env_name` with the name of the environment you wish to use.
```
conda create -n your_env_name python=3.10
conda activate your_env_name
pip install -r requirements.txt
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

## Command Usages
#### Args usages
- `--from_pretrained`: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`
- `--dataset`: Three datasets: `anli1`, `cqa`, `svamp`. To comply with OpenAI's policies and open-source licenses, we include only sample data examples of the svamp dataset in this repository. Other datasets follow the same format. Data in this format can be obtained from the OpenAI API using the [Tree of Thoughts (ToT)](https://github.com/princeton-nlp/tree-of-thought-llm) method.
  - `svamp_tree1`: Data for training the CoT reasoning capability
  - `svamp_tree2`: Data for training the self-evaluation capability
- `--model_type`:
  - `standard`: Standard finetuning (`--label_type gt`) or distillation (`--label_type llm`)
  - `task_prefix_tree1`: Training the CoT reasoning capability
  - `task_prefix_tree2`: Train the self-evaluation capability
- `--label_type`:
  - `--label_type gt`: Use GT label for training
  - `--label_type llm`: Use LLM predicted label for training
- `--alpha`: Task weight for multi-task training. Loss = alpha * label_prediction_loss + (1 - alpha) * rationale_generation_loss. Suggested value: `0.5`
- `--batch_size`: Batch size
- `--max_input_length`: Maximum input length
- `--run`: Random seed to use
- `--prompt`: Control whether to use the prompt or not. No need to use by default.
- `--num_train_branches`: The number of CoTs used for each data sample
- `--num_train_epochs`: The number of training epochs


#### Example usages
- Standard distillation:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree1 --model_type standard --label_type llm --batch_size 16 --num_train_epochs 300 --lr 5e-5 --max_input_length 1024 --run 0
```

- 1 CoT:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree1 --model_type task_prefix_tree1 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 300 --num_train_branches 1 --lr 5e-5 --max_input_length 1024 --run 0
```

- 1 CoT with Self-Evaluation:

First, learn the self-evaluation capability:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree2 --model_type task_prefix_tree2 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 150 --num_train_branches 5 --lr 5e-5 --max_input_length 1024 --run 0
```
Then learn the CoT reasoning capability:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree1 --model_type task_prefix_tree1 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 300 --num_train_branches 1 --lr 5e-5 --max_input_length 1024 --run 0 --continue_train
```

- 5 CoTs:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree1 --model_type task_prefix_tree1 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 60 --num_train_branches 5 --lr 5e-5 --max_input_length 1024 --run 0
```

- 5 CoTs with Self-Evaluation:

First, learn the self-evaluation capability:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree2 --model_type task_prefix_tree2 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 50 --num_train_branches 5 --lr 5e-5 --max_input_length 1024 --run 0
```
Then learn the CoT reasoning capability:
```python
python run.py --from_pretrained google/t5-v1_1-base --dataset svamp_tree1 --model_type task_prefix_tree1 --label_type llm --llm gpt-3.5-turbo --alpha 0.5 --batch_size 16 --num_train_epochs 80 --num_train_branches 5 --lr 5e-5 --max_input_length 1024 --run 0 --continue_train
```

The above commands are for using pseudo-labels. To use the dataset's human-annotated labels, simply change `--label_type` to `gt`.

#### Note

If an error similar to `ValueError: Failed to concatenate on axis=1 because tables don't have the same number of rows` occurs, please check the comments and range in the `data_utils.py` file.


## Cite
If you find this repository or our paper useful, please cite:
```bibtex
@inproceedings{liu2024mind,
  title={Mind’s Mirror: Distilling Self-Evaluation Capability and Comprehensive Thinking from Large Language Models},
  author={Liu, Weize and Li, Guocong and Zhang, Kai and Du, Bang and Chen, Qiyuan and Hu, Xuming and Xu, Hongxia and Chen, Jintai and Wu, Jian},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={6748--6763},
  year={2024}
}
```
