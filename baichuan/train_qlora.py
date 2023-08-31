# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 19:41
# @Author  : supinyu
# @File    : train_qlora.py
from collections import defaultdict

from loguru import logger
import os

import torch
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import HfArgumentParser, set_seed, AutoConfig, AutoTokenizer, AutoModel, DataCollatorForSeq2Seq, \
    Trainer, AutoModelForCausalLM, BitsAndBytesConfig

from arguments import ModelArguments, DataTrainingArguments, FineTuneArguments

IGNORE_INDEX = -100


class ModifiedTrainer(Trainer):

    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        save_tunable_parameters(
            self.model, os.path.join(output_dir, "chatglm-lora.pt")
        )
        self.model.save_pretrained(output_dir)


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FineTuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Setup logging

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(os.path.join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    model_name = model_args.model_name_or_path.split("/")[-1]
    logger.info("train model name is {}".format(model_name))

    # Load dataset
    data_files = {}
    data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    prefix = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："

    # Load pretrained model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    device_map = "auto"
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if model_name == "baichuan-7B" or model_name == "Baichuan-13B-Chat" or model_name == "Qwen-7B-Chat":
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch_dtype,
                                                     load_in_4bit=True,
                                                     quantization_config=BitsAndBytesConfig(
                                                         load_in_4bit=True,
                                                         bnb_4bit_compute_dtype=torch_dtype,
                                                         bnb_4bit_use_double_quant=True,
                                                         bnb_4bit_quant_type="nf4",
                                                         llm_int8_threshold=6.0,
                                                         llm_int8_has_fp16_weight=False, ),
                                                     )
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, device_map="auto")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    target_modules = find_all_linear_names(model)

    if model_name == "baichuan-7B":
        tokenizer.pad_token_id = 0

    if model_name == "Qwen-7B-Chat" or model_name == "Qwen-7B":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    if model_name == "Baichuan-13B-Chat":
        # model.supports_gradient_checkpointing = True  #节约cuda
        # model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    config = LoraConfig(r=training_args.lora_rank,
                        lora_alpha=32,
                        target_modules=target_modules,
                        lora_dropout=0.1,
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        bias="none"
                        )

    model = get_peft_model(model, config)
    #     model = model.half()
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    verify_model_dtype(model)

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    def preprocess_function_train_v1(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_train_v2(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                         max_length=data_args.max_source_length)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                         max_length=data_args.max_target_length)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_train_bai_chuan(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                query = prefix + query

                a_ids = tokenizer.encode(text=query, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length:
                    a_ids = a_ids[: data_args.max_source_length]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = a_ids + [tokenizer.bos_token_id] + b_ids + [tokenizer.eos_token_id]

                context_length = len(a_ids) + 1
                labels = [IGNORE_INDEX] * context_length + b_ids + [tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_function_train_qian_wen(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples["conversations"])):
            if examples["conversations"][i]:
                conversation = examples["conversations"][i]
                utterances = []
                for x in conversation:
                    utterances.append(x['human'])
                    utterances.append(x['assistant'])
                utterances_ids = tokenizer(utterances, add_special_tokens=False).input_ids

                input_ids = []
                target_mask = []

                for i, utterances_id in enumerate(utterances_ids):
                    input_ids += (utterances_id + [tokenizer.eos_token_id])
                    if i % 2 == 0:
                        target_mask += [-100] * (len(utterances_id) + 1)
                    else:
                        target_mask += utterances_id + [tokenizer.eos_token_id]

                assert len(input_ids) == len(target_mask)

                input_ids = input_ids[:max_seq_length]
                target_mask = target_mask[:max_seq_length]
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(target_mask)
        return model_inputs

    train_data_process_dict = {
        "chatglm2-6b": preprocess_function_train_v2,
        "chatglm-6b": preprocess_function_train_v1,
        "baichuan-7B": preprocess_function_train_bai_chuan,
        "Baichuan-13B-Chat": preprocess_function_train_bai_chuan,
        "Qwen-7B-Chat" : preprocess_function_train_bai_chuan,
        "Qwen-7B" : preprocess_function_train_qian_wen
    }

    def print_dataset_example(example):
        print(example["input_ids"])
        print(example["labels"])

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples)).shuffle()
    column_names = train_dataset.column_names
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            train_data_process_dict[model_name],
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    print_dataset_example(train_dataset[0])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Initialize our Trainer
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    train_result = trainer.train()
    model.save_pretrained(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
