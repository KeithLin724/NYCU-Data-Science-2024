import asyncio
import os
from dataclasses import dataclass

import groq
import pandas as pd
from dotenv import load_dotenv
from groq import AsyncGroq
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
import json

# from rich import print
load_dotenv()


@dataclass
class ModelSelect:
    llama2: str = "llama2-70b-4096"
    mixtral: str = "mixtral-8x7b-32768"
    gemma: str = "gemma-7b-it"


class Main:
    FILE_PATH = "./data-science-hw2-prompt-engineering/submit.csv"
    CHUNK_SIZE = 25
    JSON_TEMP = "tmp.jsonl"

    def __init__(self) -> None:
        self._table = pd.read_csv(Main.FILE_PATH).drop(columns="Unnamed: 0")
        self._client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        with open(file=Main.JSON_TEMP, mode="w") as f:
            pass

    @staticmethod
    def make_question(dict_list: dict) -> dict:
        "for model"
        result_dict = {"role": f"You are a {dict_list['task']} student"}
        dict_list.pop("task")

        result = f"Here is a question {dict_list['input']}, and this is a choose\n"
        dict_list.pop("input")
        result_choose = [f"{type_}: {text}" for type_, text in dict_list.items()]
        result_choose = "\n".join(result_choose)
        result += result_choose

        result += "\nwhich one is a answer ? please replay following this format (A) (B) (C) or (D), no need to explain"
        return result_dict | {"content": result}

    async def ask(self, role: str, question: str, model: str):
        chat_completion = await self._client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": role,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content

    async def ask_question(self, question_line: dict, id_: int, model: str):
        result = {"id": id_}  # , "target" :

        question = Main.make_question(question_line)

        response = await self.ask(question["role"], question["content"], model=model)

        return result | {"target": response}

    @staticmethod
    async def cool_down_api(chunk_pack: list):
        # wait i mins
        result = await asyncio.gather(*chunk_pack)

        with open(file=Main.JSON_TEMP, mode="a", encoding="utf-8") as f:
            for item in result:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        for i in tqdm(range(60), desc="Waiting...", unit="s"):
            await asyncio.sleep(1)

        return result

    @staticmethod
    async def gather(*task) -> list:
        "make like asyncio"

        ## cut to chunk
        chunk_task = [
            task[i : i + Main.CHUNK_SIZE] for i in range(0, len(task), Main.CHUNK_SIZE)
        ]

        tasks_result = [
            await Main.cool_down_api(chunk_pack)
            for chunk_pack in tqdm(chunk_task, desc="Asking...", unit="chunk")
        ]
        # add to result
        tasks_result = sum(tasks_result, [])
        return tasks_result

    async def run(self):
        model = ModelSelect.llama2

        question_bank = self._table.to_dict("records")

        tasks = [
            self.ask_question(question_line, index, model)
            for index, question_line in enumerate(question_bank)
        ]

        result = await Main.gather(*tasks)
        df_result = pd.DataFrame(result)
        df_result.to_csv(f"{model}_pre.csv")

        return


if __name__ == "__main__":
    asyncio.run(Main().run())
