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
import click
import random as rd


load_dotenv()


@dataclass
class ModelSelect:
    llama2: str = "llama2-70b-4096"
    mixtral: str = "mixtral-8x7b-32768"
    gemma: str = "gemma-7b-it"


class Main:
    # FILE_PATH = "./data-science-hw2-prompt-engineering/submit.csv"
    CHUNK_SIZE = 25
    JSON_TEMP = "tmp.jsonl"

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._table = pd.read_csv(self._file_path)
        self._table = self._table.rename(columns={"Unnamed: 0": "ID"})
        self._table_true = pd.read_csv("./submit_true.csv")

        self._table.loc[:, "answer"] = self._table_true.loc[:, "answer"]

        self._client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

        # make a temp file
        with open(file=Main.JSON_TEMP, mode="w") as f:
            pass

    @staticmethod
    def make_question(dict_list: dict) -> dict:
        "for model"
        result_dict = {"role": f"You are a {dict_list['task']} professor"}
        dict_list.pop("task")

        result = f"Here is a question {dict_list['input']}, and this is a choose\n"
        dict_list.pop("input")
        result_choose = [f"{type_}: {text}" for type_, text in dict_list.items()]
        result_choose = "\n".join(result_choose)
        result += result_choose

        result += f"\nThe correct answer is {dict_list['answer']}. why? Please provide a step-by-step explanation of your solution, please replay the correct answer following this format (A) (B) (C) or (D)"
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

    async def ask_question(self, question_line: dict, model: str):
        result = {"ID": question_line["ID"]}  # , "target" :

        question = Main.make_question(question_line)

        response = await self.ask(question["role"], question["content"], model=model)

        return result | {"target": response}

    @staticmethod
    async def cool_down_api(chunk_pack: list):
        # wait i mins
        result = await asyncio.gather(*chunk_pack)

        # save file
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

    @staticmethod
    def handle_ans(row: dict) -> dict:

        mapping = {
            ("(A)", "A:", "A.", "A)"): "A",
            ("(B)", "B:", "B.", "B)"): "B",
            ("(C)", "C:", "C.", "C)"): "C",
            ("(D)", "D:", "D.", "D)"): "D",
        }

        def change_is(str_part: str) -> str:
            for check, change in mapping.items():
                if check in str_part:
                    return change

            return None

        target = row["target"]
        target = target.split("\n")

        target = [change_is(str_part) for str_part in target]
        target = [item for item in target if item is not None]

        row["target"] = (
            target[0] if len(target) != 0 else rd.choice(list(mapping.values()))
        )
        return row

    @staticmethod
    def model_select(model_id: int) -> str:
        model_list = [ModelSelect.llama2, ModelSelect.mixtral, ModelSelect.gemma]
        return model_list[model_id - 1]

    async def run(self, model_id: int):
        model = Main.model_select(model_id)

        question_bank = self._table.to_dict("records")

        # print(question_bank)

        tasks = [
            self.ask_question(question_line, model) for question_line in question_bank
        ]

        result = await Main.gather(*tasks)
        df_result = pd.DataFrame(result)
        df_result.to_csv(f"{model}_pre.csv", index=False)

        df_ans = df_result.apply(Main.handle_ans, axis=1)
        df_ans.to_csv(f"{model}_ans.csv", index=False)

        return


@click.command()
@click.option(
    "-f",
    "--file_path",
    help="file name",
    required=True,
    type=str,
)
@click.option(
    "-id",
    "--model_id",
    default=1,
    type=int,
    help="Select Model to Run [1: llama2 ,2: mixtral, 3: gemma]",
    show_default=True,
)
def main(file_path: str, model_id: int) -> None:
    "run the app"
    asyncio.run(Main(file_path).run(model_id))
    return


if __name__ == "__main__":
    main()
