import json
from datasets import load_dataset
from transformers import AutoTokenizer


def sample_mapping_fn(sample, 
                      llm_tokenizer, 
                      max_input_length:int,
                      instruct_prompt:str = "", 
                      sys_prompt:str = None,
                      reward_model_style: str = None):
    """
    Batch-processing version of sample_mapping_fn.

    Args:
        sample (dict): Batch of samples, e.g.
            {
              "data":     [json_str1, json_str2, ...],
              "question": [q1, q2, ...],
              "label":    [lbl1, lbl2, ...]
            }
        llm_tokenizer: Tokenizer with apply_chat_template().
        max_input_length (int): Max tokens for the model input.
        instruct_prompt (str): Instruction to be added to the prompt.
        sys_prompt (str): System prompt to be added at the beginning of the message.
        reward_model_style (str): Style for the reward model, e.g., "gt"

    Returns:
        dict: {
            "message": [ [ {role,content}, ... ],  # one list per example
                         ... ],
            "label":   [ lbl1, lbl2, ... ]
        }
    """
    # 计算剩余可用 token 长度
    #    先算仅 question+instruction 模板需要多少 token
    question = sample["question"]
    if isinstance(question, list):
        question = question[0]
    nodata_messages = [
        {"role": "user", "content": question + instruct_prompt}
    ]
    nodata_tokens = llm_tokenizer.apply_chat_template(nodata_messages)
    max_data_tokens = max_input_length - len(nodata_tokens)

    message_batch = []
    data_source_batch = sample["task_id"]
    reward_model_batch = []

    # 逐条处理
    for data_json, label in zip(
        sample["data"], sample["label"]
    ):
        # 1) 载入 data 字典
        data_dict = json.loads(data_json)

        # 2) 把 data_dict 转成字符串片段，截断到 max_data_tokens
        data_str = _turn_data_dict_to_str(
            data_dict, max_data_tokens, llm_tokenizer
        )
        
        # 3) 拼接最终 prompt
        prompt = data_str + question + instruct_prompt

        # 4) 按 chat 格式包装成 list-of-messages
        message = [{"role": "user", "content": prompt}]
        if sys_prompt is not None:
            message = [{"role": "system", "content": sys_prompt}] + message
        message_batch.append(message)

        # 5) 保存标签
        reward_model_batch.append({"ground_truth": label, "style": reward_model_style})
    

    return {
        "message": message_batch,
        "data_source": data_source_batch,
        "reward_model": reward_model_batch,
    }


def _turn_data_dict_to_str(data_dict: dict, max_data_tokens_length: int, llm_tokenizer, data_prefix: str = "## Data\n"):
    """
    Convert a dictionary to a string.
    This function will calculate the maximum length of the data and return the string representation of the data.
    Args:
        data_dict (dict): The dictionary to be converted to a string.
        max_data_tokens_length (int): The maximum available data tokens length.
        llm_tokenizer: The tokenizer used to encode the data.
        data_prefix (str): The prefix to add before the data.
    Returns:
        str: The string representation of the dictionary.
    """
    if type(data_dict) is dict:
        # Extract the static events
        static_sentences = data_dict.pop("Static", [])
        static_sentence = "".join(static_sentences) if static_sentences else ""
        static_tokens = llm_tokenizer.encode(static_sentence, add_special_tokens=False) if static_sentence else []
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(static_tokens) + len(data_prefix_tokens)

        sentences = []
        tokens = []

        for time, str_events in reversed(data_dict.items()):
            if not str_events:
                continue
            if type(str_events) is str:
                str_events = [str_events]
            time_sentence = "".join(str_events)
            time_tokens = llm_tokenizer.encode(time_sentence, add_special_tokens=False)
            if len(tokens) + len(time_tokens) > max_data_tokens_length:
                break
            # Extend time_tokens at the beginning of the tokens list
            tokens = time_tokens + tokens
            sentences = str_events + sentences

        sentences = [data_prefix, static_sentence] + sentences
        input_prompt = "".join(sentences)
    elif type(data_dict) is str:
        data_prefix_tokens = llm_tokenizer.encode(data_prefix, add_special_tokens=False)
        max_data_tokens_length -= len(data_prefix_tokens)
        data_tokens = llm_tokenizer.encode(data_dict, add_special_tokens=False)
        if len(data_tokens) > max_data_tokens_length:
            data_tokens = data_tokens[-max_data_tokens_length:]
            input_prompt = llm_tokenizer.decode(data_tokens)
        else:
            input_prompt = data_dict
    return input_prompt


def map_gender_to_index(gender:str):
    if gender in ["F"]:
        return 0
    elif gender in ["M"]:
        return 1
    else:
        raise ValueError(f"invalid gender: {gender}")
    

def map_race_to_index(race:str):
    """
    map string race into index
    needed to be updatec
    """
    race = race.lower()
    if "white" in race:
        return 0
    elif "black" in race:
        return 1
    elif "hispanic" in race:
        return 2
    elif "asian" in race:
        return 3
    elif "other" in race or "hawaiian" in race or "south american" in race:
        return 4
    else: # unknown case
        return 5
    

if __name__ == "__main__":
    import os
    BASE_DIR = "/mnt/hdfs/seed_zhiyao/Med/EHR_QA_v2/EHRSHOT"
    OUTPUT_DIR = "/mnt/hdfs/seed_zhiyao/Med/EHR_QA_v2/EHRSHOT_rlvr"
    task_types = ["guo_readmission", "new_pancan"]   # can extend with more task types later
    splits = ["held_out", "train", "tuning"]              # can extend with more dataset splits later
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task_type in task_types:
        for split in splits:
            data_path = os.path.join(BASE_DIR, f"{task_type}/nl/{split}.parquet")
            data_files = {split: data_path}

            dataset = load_dataset(
                "parquet",
                data_files=data_files,
                columns=["task_id", "label", "data", "question"]
            )

            llm_tokenizer = AutoTokenizer.from_pretrained(
                "/mnt/hdfs/seed_zhiyao/pretrained_models_alphaseed/Qwen2.5-Coder-7B"
            )

            dataset = dataset.map(
                sample_mapping_fn,
                fn_kwargs={
                    "instruct_prompt": " Let's think step by step.",
                    "llm_tokenizer": llm_tokenizer,
                    "max_input_length": 8 * 1024,
                    "reward_model_style": task_type
                },
                batched=True,
                num_proc=8
            )

            print(f"=== {task_type} | {split} ===")
            print(dataset[split]["message"][0])
            print(dataset[split]["data_source"][0])
            print(dataset[split]["reward_model"][0])
            os.makedirs(os.path.join(OUTPUT_DIR, f"{task_type}/nl"), exist_ok=True)
            df = dataset[split].to_pandas()
            df.to_parquet(os.path.join(OUTPUT_DIR, f"{task_type}/nl/{split}.parquet"), index=False)

    