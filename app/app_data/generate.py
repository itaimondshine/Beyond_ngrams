import csv
import json
import logging
import multiprocessing as mp

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from timeout_decorator import timeout_decorator
import time
import datasets
from typing import Any, Dict, List, NewType, Optional, Union
from datasets import Dataset
import backoff
import google.generativeai as genai
import numpy as np
import requests
import yaml
from datasets import load_dataset
from easygoogletranslate import EasyGoogleTranslate
from tqdm import tqdm
from yaml.loader import SafeLoader
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

logger = logging.Logger("Xlsum_task")
LANGUAGE_TO_SUFFIX = {
    "chinese_simplified": "zh-CN",
    "french": "fr",
    "portuguese": "pt",
    "english": "en",
    "arabic": "ar",
    "ukrainian": "uk",
    "indonesian": "id",
    "hindi": "hi",
    "indonesian": "id",
    "amharic": "am",
    "bengali": "bn",
    "burmese": "my",
    "uzbek": "uz",
    "chinese": "zh-CN",
    "nepali": "ne",
    "japanese": "ja",
    "spanish": "es",
    "turkish": "tr",
    "persian": "fa",
    "azerbaijani": "az",
    "korean": "ko",
    "yoruba": "yo",
    "vietnamese": "vi",
}

PARAMS = NewType("PARAMS", Dict[str, Any])


def _translate_example(
    example: Dict[str, str], src_language: str, target_language: str
):
    translator = EasyGoogleTranslate(
        source_language=LANGUAGE_TO_SUFFIX[src_language],
        target_language=LANGUAGE_TO_SUFFIX[target_language],
        timeout=30,
    )
    try:
        return {
            "text": translator.translate(example["text"]),
            "summary": translator.translate(example["summary"]),
        }
    except Exception as e:
        print(e)


def choose_few_shot_examples(
    train_dataset: Dataset,
    few_shot_size: int,
    context: List[str],
    selection_criteria: str,
    lang: str,
) -> List[Dict[str, Union[str, int]]]:
    """Selects few-shot examples from training datasets

    Args:
        train_dataset (Dataset): Training Dataset
        few_shot_size (int): Number of few-shot examples
        selection_criteria (few_shot_selection): How to select few-shot examples. Choices: [random, first_k]

    Returns:
        List[Dict[str, Union[str, int]]]: Selected examples
    """
    selected_examples = []

    example_idxs = []
    if selection_criteria == "first_k":
        example_idxs = list(range(few_shot_size))
    elif selection_criteria == "random":
        example_idxs = (
            np.random.choice(len(train_dataset), size=few_shot_size, replace=True)
            .astype(int)
            .tolist()
        )

    ic_examples = [
        {"text": train_dataset[idx]["text"], "summary": train_dataset[idx]["summary"]}
        for idx in example_idxs
    ]

    for idx, ic_language in enumerate(context):
        (
            selected_examples.append(ic_examples[idx])
            if ic_language == lang
            else (
                selected_examples.append(
                    _translate_example(
                        example=ic_examples[idx],
                        src_language=lang,
                        target_language=ic_language,
                    )
                )
            )
        )

    return selected_examples


def read_parameters(args_path) -> PARAMS:
    with open(args_path) as f:
        args = yaml.load(f, Loader=SafeLoader)
    return args


def get_key(key_path):
    with open(key_path) as f:
        key = f.read().split("\n")[0]
    return key


def load_xlsum_data(lang, split, limit):
    """Loads the xlsum dataset"""
    data = pd.read_csv(
        f"/Users/itaimondshine/PycharmProjects/NLP/eval_metrics/app/app_data/xlsum/{lang}.csv"
    )
    data_in_dics = [doc.to_dict() for _, doc in data.iterrows()]
    return data_in_dics[0:400]


def gemini_completion(prompt):
    # Define the endpoint URL
    genai.configure(api_key="AIzaSyBgja1_jiDdyVsIA0KBamDdgECkZ7r32y8")
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    return model.generate_content(prompt).text


#
# def gemini_completion(prompt, api_key="AIzaSyCSvECR2K_ca3QcMBcCHbxMzBpZe3y82iI"):
#     url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}'
#     headers = {
#         'Content-Type': 'application/json'
#     }
#     data = {
#         'contents': [
#             {
#                 'parts': [
#                     {
#                         'text': prompt
#                     }
#                 ]
#             }
#         ]
#     }
#
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return {'error': response.text}


def gpt3x_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "chatgpt",
    # run_details: Any = {},
    # num_evals_per_sec: int = 2,
    # **model_params,
) -> str:
    import os
    import openai

    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    def get_entities_chatGPT(final_prompt):
        response = openai.ChatCompletion.create(
            engine="gpt35-16k",
            temperature=0,
            messages=[{"role": "user", "content": final_prompt}],
        )
        return response["choices"][0]["message"]["content"]

    return get_entities_chatGPT(final_prompt=prompt)


@backoff.on_exception(
    backoff.expo, max_tries=10, exception=requests.exceptions.RequestException
)
def llama_completion(prompt):
    url = "https://api.together.xyz/v1/chat/completions"

    # Define your Together API key
    together_api_key = "eba6f0c1eef11bef1d16232c39a4797b8bd50b2253bba4778c55836c4bddc275"  # Replace with your actual API key

    # Define the request payload
    payload = {
        "temperature": 0,
        "model": "togethercomputer/Llama-2-7B-32K-Instruct",
        "messages": [{"role": "user", "content": f"{prompt}"}],
    }

    # Define request headers
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json",
    }

    # Send POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check response status
    if response.status_code == 200:
        # Print the response content (API output)
        return response.json()["choices"][0]["message"]["content"]
    else:
        # Print error message if request fails
        print(f"Error: {response.status_code} - {response.text}")


@backoff.on_exception(
    backoff.expo, max_tries=10, exception=requests.exceptions.RequestException
)
def mixtral_completion(prompt):
    url = "https://api.together.xyz/v1/chat/completions"

    # Define your Together API key
    together_api_key = "851cfc39f3d7a246a2342259f5f6fbba4721c6002123365fba2254c9c9c424ad"  # Replace with your actual API key

    # Define the request payload
    payload = {
        "temperature": 0,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": f"{prompt}"}],
    }

    # Define request headers
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json",
    }

    # Send POST request
    response = requests.post(url, json=payload, headers=headers)

    # Check response status
    if response.status_code == 200:
        # Print the response content (API output)
        return response.json()["choices"][0]["message"]["content"]
    else:
        # Print error message if request fails
        print(f"Error: {response.status_code} - {response.text}")


def construct_prompt(
    instruction: str,
    test_example: dict,
    ic_examples: List[dict],
    zero_shot: bool,
    lang: str,
    config: Dict[str, str],
):

    example_prompt = PromptTemplate(
        input_variables=["summary", "text"], template="Text: {text}\nSummary: {summary}"
    )

    zero_shot_template = f"""{instruction}""" + "\n {text} " ""

    prompt = (
        FewShotPromptTemplate(
            examples=ic_examples,
            prefix=instruction,
            example_prompt=example_prompt,
            suffix="<Text>: {text}",
            input_variables=["text"],
        )
        if not zero_shot
        else PromptTemplate(input_variables=["text"], template=zero_shot_template)
    )

    label = test_example["summary"]
    if config["input"] != lang:
        test_example = _translate_example(
            example=test_example, src_language=lang, target_language=config["input"]
        )

    return prompt.format(text=test_example["text"]), label


def dump_metrics(
    lang: str,
    config: Dict[str, str],
    r1: float,
    r2: float,
    rL: float,
    metric_logger_path: str,
):
    # Check if the metric logger file exists
    file_exists = os.path.exists(metric_logger_path)

    # Open the CSV file in append mode
    with open(metric_logger_path, "a", newline="") as f:
        csvwriter = csv.writer(f, delimiter=",")

        # Write header row if the file is newly created
        if not file_exists:
            header = [
                "Language",
                "Prefix",
                "Input",
                "Context",
                "Output",
                "R1",
                "R2",
                "RL",
            ]
            csvwriter.writerow(header)

        csvwriter.writerow(
            [
                lang,
                config["prefix"],
                config["input"],
                config["context"][0],
                config["output"],
                r1,
                r2,
                rL,
            ]
        )


def dump_predictions(idx, response, label, response_logger_file):
    obj = {"q_idx": idx, "prediction": response, "label": label}
    with open(response_logger_file, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_rouge(scorer, pred, label):
    score = scorer.score(pred, label)
    return score["rouge1"], score["rouge2"], score["rougeL"]


def _translate_instruction(basic_instruction: str, target_language: str) -> str:
    translator = EasyGoogleTranslate(
        source_language="en",
        target_language=LANGUAGE_TO_SUFFIX[target_language],
        timeout=50,
    )
    return translator.translate(basic_instruction)


def _translate_example(
    example: Dict[str, str], src_language: str, target_language: str
):
    translator = EasyGoogleTranslate(
        source_language=LANGUAGE_TO_SUFFIX[src_language],
        target_language=LANGUAGE_TO_SUFFIX[target_language],
        timeout=30,
    )
    try:
        return {
            "text": translator.translate(example["text"][:2000])
            + translator.translate(example["text"][2000:4000])
            + translator.translate(example["text"][4000:6000]),
            "summary": translator.translate(example["summary"]),
        }
    except Exception as e:
        print(example["text"])
        print(example["summary"])
        print(e)


def _translate_prediction_to_output_language(
    prediction: str, prediction_language: str, output_language: str
) -> str:
    translator = EasyGoogleTranslate(
        source_language=LANGUAGE_TO_SUFFIX[prediction_language],
        target_language=LANGUAGE_TO_SUFFIX[output_language],
        timeout=10,
    )
    return translator.translate(prediction)


def create_instruction(lang: str, expected_output: str):
    basic_instruction = f"""You are a good summarizer!"
         Your task is to generate a short summary of a text in {expected_output} language.
         Summarize the text below, delimited by triple backticks, in at 5 sentences!
         Make sure the length is no more than 5 sentences, if it's more - make it shorter"""
    return (
        basic_instruction
        if lang == "english"
        else _translate_instruction(basic_instruction, target_language=lang)
    )


def run_one_configuration_paralle(params: Optional[PARAMS] = None, zero: bool = True):
    if not params:
        params = read_parameters("parameters.yaml")

    lang = params["selected_language"]
    config = params["config"]
    zero = len(config["context"]) == 0

    if zero:
        config_header = f"{config['input']}_{config['prefix']}_zero_{config['output']}"
    else:
        config_header = f"{config['input']}_{config['prefix']}_{config['context'][0]}_{config['output']}"

    test_data = load_xlsum_data(lang=lang, split="test", limit=params["limit"])

    # Initialize multiprocessing pos
    num_processes = mp.cpu_count()  # Use number of available CPU cores
    pool = mp.Pool(processes=1)

    # Iterate over test_data using tqdm for progress tracking
    for _, test_example in tqdm(enumerate(test_data), total=len(test_data)):
        # Apply asynchronous processing of each test example
        pool.apply_async(
            process_test_example,
            args=(
                test_data,
                config_header,
                test_example["q_idx"],
                test_example,
                config,
                zero,
                lang,
                params,
            ),
        )

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


import subprocess


@timeout_decorator.timeout(5)
def get_prediction(prompt, endpoint_id, project_id, location="us-east1"):
    # Get the access token using gcloud
    access_token = "ya29.c.c0ASRK0Ga__3PZLS7tbS8Ye5fB3gOwOp3-oSdWzZ0RSFJEq_GRvvOTtzXNA1Jq30FVDhk31fv82McjxYbqnQj1qdjz0GvTdgkSiCDtx-TOS4j4nLbpbJgddg2_Olo3wvDuekEj93FrOnqgPHGzxzeZTV5_mj8y2RNxSJZDWwEIWKayOE1ozt6vVJxYMKWzMNRPuaYSrSyHgkqzvzDZFpLo64MB2Fg8DHK8uJaja7j7hiRxsZmsP_Z9Rs6RqJGV1jw1r2zV_V5y6UDbmPgZI9YuXl_lTCJXG2pcbCGc-DMIa1ggVwbll4-Ek8-oAXEhrYyq6JrS2y4xjkUS3CWXcit5q84Lm9VhPEhqXv1_g4t4SiQeVXBph1V5_7LXMP1hTAE391KvdOd7erunO_YQujkU5baaVSFygOUdc2zi9sd9bU-9Fdhk2w8B3BeIo9Fg7zJc0g4nhIWtMZ9zSMBoIvX9QmW0vOss4g5OSZeb_dl_uvZ4hVB3XJXe2QhhzY6pQ-wcuMIaxVjqQBFUouWndib3Oo36zQ9trYkkF9_SJblQbogxj9oZFzgInmYon8Ftsl66UrYJs7-zl7dnMXVXZnw1uQ7muayxUjwk6yVxz4o0f8qx8iJUWzInMwulJblY3srhV8Wa8URyl2pudauoz3iWbJJ12VZX-S2B1p0bcS1f1wvy-daX12Qbx8f__-vV9Bi6Wp345r-wexzId1JtOVxzJ83yabi0FkyMJkvRIjX9nFc1JzWjWOOMJZJSnMs9zdrZxveS6pQI_R2cv0xdRgj2rf7Bkm4spbJyMoM4sbyhMxmdWhyu5rU2Rt6I37SrYb_wt7nWdtynZ68B1Vp-p0w49nMJkv4R9J9mlJnWb0zbY9VWIJwXQbVZQv7nItFgRSM01x1lqWax-UfIjvwV9X1QcqgoeRVo-3V1t5XZ4u801ritdynWIB6y6XhVv-j9cafVqO8R_M_ccWSfjaklumSw1dkwSlr1ubi15qbMFRj_Fy5k2xJj0c7e7Bu"

    # Prepare the URL
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"

    # Prepare the input data
    input_data = {"instances": [{"inputs": prompt}]}

    # Prepare the headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=input_data)

    # Check the response and return the result
    if response.status_code == 200:
        return response.json()["predictions"]  # Return the JSON response from the model
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


def mt0_completion(prompt):
    checkpoint = "bigscience/mt0-xxl-mt"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))


def process_test_example(
    test_data, config_header, idx, test_example, config, zero_shot, lang, params
):
    try:
        instruction = create_instruction(
            lang=config["prefix"], expected_output=config["output"]
        )
        text_example = {
            "text": test_example["text"],
            "summary": test_example["label"],
        }

        ic_examples = []
        if not zero_shot:
            ic_examples = choose_few_shot_examples(
                train_dataset=test_data,
                few_shot_size=len(config["context"]),
                context=config["context"],
                selection_criteria="random",
                lang=params["selected_language"],
            )

        prompt, label = construct_prompt(
            instruction=instruction,
            test_example=text_example,
            ic_examples=ic_examples,
            zero_shot=zero_shot,
            lang=lang,
            config=config,
        )

        print(prompt)
        # pred = get_prediction(prompt=prompt, endpoint_id=7327255438662041600, project_id=16514800572)
        time.sleep(1)
        pred = gpt3x_completion(prompt)
        number_of_sentences = pred.split(".")
        print(len(number_of_sentences))

        logger.info("Saving prediction to persistent volume")
        os.makedirs(
            f"{params['response_logger_root']}/{params['model']}/{lang}", exist_ok=True
        )
        dump_predictions(
            idx=idx,
            response=pred,
            label=label,
            response_logger_file=f"/Users/itaimondshine/PycharmProjects/NLP/eval_metrics/app/app_data/data/{lang}/{params['model']}.csv",
        )

    except Exception as e:
        print(e)


def choose_few_shot_examples(
    train_dataset: Dataset,
    few_shot_size: int,
    context: List[str],
    selection_criteria: str,
    lang: str,
) -> List[Dict[str, Union[str, int]]]:
    """Selects few-shot examples from training datasets

    Args:
        train_dataset (Dataset): Training Dataset
        few_shot_size (int): Number of few-shot examples
        selection_criteria (few_shot_selection): How to select few-shot examples. Choices: [random, first_k]

    Returns:
        List[Dict[str, Union[str, int]]]: Selected examples
    """
    selected_examples = []

    example_idxs = []
    if selection_criteria == "first_k":
        example_idxs = list(range(few_shot_size))
    elif selection_criteria == "random":
        example_idxs = (
            np.random.choice(len(train_dataset), size=few_shot_size, replace=True)
            .astype(int)
            .tolist()
        )

    ic_examples = [
        {"text": train_dataset[idx]["text"], "summary": train_dataset[idx]["summary"]}
        for idx in example_idxs
    ]

    for idx, ic_language in enumerate(context):
        (
            selected_examples.append(ic_examples[idx])
            if ic_language == lang
            else (
                selected_examples.append(
                    _translate_example(
                        example=ic_examples[idx],
                        src_language=lang,
                        target_language=ic_language,
                    )
                )
            )
        )

    return selected_examples


if __name__ == "__main__":
    run_one_configuration_paralle()
