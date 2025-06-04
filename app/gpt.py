import os
from typing import Union, List, Dict

import openai


def gpt3x_completion(
    prompt: Union[str, List[Dict[str, str]]],
) -> str:
    openai.api_key = os.environ.get("OPEN_AI_KEY")
    openai.api_type = "azure"

    def get_entities_chatGPT(final_prompt):
        response = openai.ChatCompletion.create(
            engine="gpt35-16k",
            temperature=0,
            messages=[{"role": "user", "content": final_prompt}],
        )
        return response["choices"][0]["message"]["content"]

    return get_entities_chatGPT(final_prompt=prompt)
