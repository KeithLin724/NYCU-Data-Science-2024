from datasets import load_dataset
import os
import pandas as pd
import csv
import numpy as np
import evaluate
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    # trainer
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch.nn.utils.prune as prune
from datasets import load_dataset


class TALib:
    CHECKPOINT = "t5small_TextSummarization/"  # released full model path
    TK_ckpt = "t5-small"

    def __init__(self):
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.load_data()

        return

    def load_data(self):
        self.billsum, self.billsum_test = (
            load_dataset("billsum", split="train"),
            load_dataset("billsum", split="test"),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(TALib.TK_ckpt)

        preprocess_function = TALib.preprocess_function_pass_tokenizer(self.tokenizer)

        self.billsum = self.billsum.train_test_split(test_size=0.2)

        self.tokenized_billsum = self.billsum.map(preprocess_function, batched=True)
        self.tokenized_billsum_test = self.billsum_test.map(
            preprocess_function, batched=True
        )

        return

    def get_trainer(
        self,
        model: T5ForConditionalGeneration,
        num_train_epochs: int = 4,
        batch_size: int = 2,
        output_dir: str = "TA_billsum_model",
    ):

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=TALib.CHECKPOINT
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,  # Assuming you still want weight decay as it wasn't mentioned to remove
            save_total_limit=3,  # Assuming to maintain the save limit as before
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="linear",
            seed=42,
            fp16=True,  # You mentioned "Native AMP" for mixed precision training which is generally enabled by setting fp16=True in Transformers
            logging_steps=10,  # Assuming to keep the logging frequency as before
            predict_with_generate=True,
        )

        compute_metrics = TALib.compute_metrics_pass_tokenizer(self.tokenizer)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_billsum["train"],
            eval_dataset=self.tokenized_billsum["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        return trainer

    def predict_and_dump(self, trainer: Seq2SeqTrainer, filename: str):
        results = trainer.predict(self.tokenized_billsum_test)
        decoded_prediction = self.tokenizer.batch_decode(
            results[0], skip_special_tokens=True
        )

        TALib.dump_to_kaggle_format(decoded_prediction, filename)

        return results

    # static
    @staticmethod
    def preprocess_function_pass_tokenizer(tokenizer):
        prefix = "summarize: "

        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["text"]]
            model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

            labels = tokenizer(
                text_target=examples["summary"], max_length=128, truncation=True
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return preprocess_function

    @staticmethod
    def show_param_ratio(model) -> float:
        num_param = sum(param.numel() for param in model.parameters())

        num_mask = sum(
            (param == 0).sum()
            for name, param in model.named_buffers()
            if "mask" in name
        )

        res = (num_param - num_mask) / num_param

        if isinstance(res, torch.Tensor):
            res = res.item()

        return res

    def to_ta_kaggle_format(data: list[str]) -> pd.DataFrame:

        func_item, head_columns, columns = (
            (
                (lambda item: item),
                ["ID", "Predict"],
                "Predict",
            )  # prediction
            if isinstance(data, list)
            else (
                (lambda item: item["summary"]),
                ["ID", "Label"],
                "Label",
            )
        )

        df_results = pd.DataFrame(columns=head_columns)

        for i, item in enumerate(data):
            item = func_item(item)
            # Escape quotes by replacing "," with "."
            summary_escaped = item.replace(",", ".")

            # Create a new row DataFrame and append it
            new_row = pd.DataFrame({"ID": [i], columns: [summary_escaped]})
            df_results = pd.concat([df_results, new_row], ignore_index=True)

        return df_results

    @staticmethod
    def dump_to_kaggle_format(decoded_prediction: list[str], filename: str):
        df_results = TALib.to_ta_kaggle_format(decoded_prediction)

        # Function to escape double quotes and handle newlines
        def escape_special_characters(text):
            return text.replace('"', '""').replace("\n", " ")

        # Apply escaping to the 'Summary' column
        df_results["Predict"] = df_results["Predict"].apply(escape_special_characters)

        df_results.to_csv(
            filename,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
        )

        return df_results

    @staticmethod
    def calculate_lcs(X: list, Y: list):
        """
        Helper function to calculate the longest common subsequence of sequences X and Y.
        """
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[m][n]

    @staticmethod
    def score(
        solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
    ) -> float:
        """
        Computes the ROUGE-Lsum score based on the longest common subsequence summed over all sentences in the summaries.

        Args:
        solution (pd.DataFrame): The DataFrame containing the correct summaries.
        submission (pd.DataFrame): The DataFrame containing participant's predicted summaries.
        row_id_column_name (str): The column name for the row ID in both DataFrames.

        Returns:
        float: The mean ROUGE-Lsum score across all predictions.
        """
        # Ensure indices for proper alignment
        solution.set_index(row_id_column_name, inplace=True)
        submission.set_index(row_id_column_name, inplace=True)

        total_score = 0

        for idx in solution.index:
            if idx not in submission.index:
                # raise ParticipantVisibleError(f"Missing prediction for ID {idx}.")
                raise ValueError(f"Missing prediction for ID {idx}.")

            ref_summary = solution.loc[idx, "Label"]
            pred_summary = submission.loc[idx, "Predict"]

            # Tokenize sentences
            ref_sentences = ref_summary.split(".")
            pred_sentences = pred_summary.split(".")

            # Calculate LCS for each sentence pair
            lcs_sum = 0
            for ref_sent in ref_sentences:
                ref_tokens = ref_sent.strip().lower().split()
                best_lcs = 0
                for pred_sent in pred_sentences:
                    pred_tokens = pred_sent.strip().lower().split()
                    lcs_length = TALib.calculate_lcs(ref_tokens, pred_tokens)
                    best_lcs = max(best_lcs, lcs_length)
                lcs_sum += best_lcs

            # Calculate ROUGE-L for the current pair of summaries
            ref_length = sum(len(sent.strip().split()) for sent in ref_sentences)
            if ref_length > 0:
                rouge_l = lcs_sum / ref_length
            else:
                rouge_l = 0
            total_score += rouge_l

        # Compute the average ROUGE-L score across all submissions
        mean_rouge_lsum = total_score / len(solution)

        return mean_rouge_lsum

    @staticmethod
    def run_score(predict, label):

        df_predict = TALib.to_ta_kaggle_format(predict)
        df_label = TALib.to_ta_kaggle_format(label)

        return TALib.score(
            solution=df_label,
            submission=df_predict,
            row_id_column_name="ID",
        )

    @staticmethod
    def compute_metrics_pass_tokenizer(tokenizer):
        rouge = evaluate.load("rouge")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            result = rouge.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )

            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
            ]
            result["gen_len"] = np.mean(prediction_lens)

            return {k: round(v, 4) for k, v in result.items()}

        return compute_metrics

    @staticmethod
    def save_model(model: T5ForConditionalGeneration, folder_path: str):
        "save_ta_format"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model.save_pretrained(folder_path, from_pt=True)
        torch.save(model.state_dict(), f"{folder_path}/model_state_dict.pth")

        return

    @staticmethod
    def load_model(folder_path: str) -> T5ForConditionalGeneration:
        "load_ta_format"
        if not os.path.exists(folder_path):
            raise ValueError("folder is not found")

        model = AutoModelForSeq2SeqLM.from_pretrained(folder_path)

        # Apply prune.identity to the layers that were pruned

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                # Check the layer type as per your model's pruned layers
                prune.identity(module, "weight")

        model.load_state_dict(torch.load(f"{folder_path}/model_state_dict.pth"))

        return model
