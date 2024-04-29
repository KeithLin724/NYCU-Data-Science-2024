from datasets import load_dataset
import os
import pandas as pd
import csv
import numpy as np
import evaluate


class TALib:
    CHECKPOINT = "t5small_TextSummarization/"  # released full model path
    TK_ckpt = "t5-small"

    def __init__(self):
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        return

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

        return (num_param - num_mask) / num_param

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
            "full_model_sample_submission.csv",
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
