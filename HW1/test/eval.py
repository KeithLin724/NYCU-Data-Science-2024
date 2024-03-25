import argparse
import glob
import json
import os
from itertools import combinations


class Checker:
    def __init__(self):
        # The most recent path that is loaded by `load_jsonl` or `load_json`.
        self.path = None

    def check(self, *args, **kwargs):
        try:
            return self.check_impl(*args, **kwargs)
        except FileNotFoundError:
            if self.get_filename().endswith('.json'):
                filename_only = os.path.splitext(self.get_filename())[0]
            else:
                filename_only = 'crawl'
            stderr_path = os.path.join(
                self.get_dirname(), f"{filename_only}.stderr")
            if os.path.exists(stderr_path):
                with open(stderr_path, 'r') as f:
                    error = f.read()
            else:
                error = ""
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Your output file {self.get_filename()} is not "
                    f"found. This may be caused by: 1) {{student_id}}.py is "
                    f"not found 2) Incorrect output filename. 3) Import error. "
                    f"4) Exceeding time limit. "
                    f"The following error message is collected from your "
                    f"standard error output (may be empty): "
                ) + error
            }
        except KeyError as e:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] The key {e.args[0]} is not found in your "
                    f"output file {self.get_filename()}.")
            }
        except json.decoder.JSONDecodeError:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Your output file {self.get_filename()} is not "
                    f"a valid `JSON`/`JSON Lines` file.")
            }
        except (ValueError, TypeError) as e:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Your output file {self.get_filename()} is not "
                    f"a valid file. This may be caused by incorrect variable "
                    f"type. {e}")
            }
        except Exception:
            assert False, (
                "THIS MUST NOT HAPPEN. Please contact TA if you see this "
                "message.")

    def load_jsonl(self, answer_dir, output_dir):
        def read(path):
            self.path = path
            data = set()
            with open(path) as f:
                lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Each line of {path} must be a JSON object.")
                for key in ['date', 'title', 'url']:
                    if not isinstance(item[key], str):
                        raise ValueError(
                            f"Type of `{key}` must be str, not "
                            f"{type(item[key])}.")
                data.add((
                    item['date'],
                    item['title'],
                    item['url'].split('/')[-1],  # article_id
                ))
            return data

        return (
            read(os.path.join(answer_dir, 'articles.jsonl')),
            read(os.path.join(answer_dir, 'popular_articles.jsonl')),
            read(os.path.join(output_dir, 'articles.jsonl')),
            read(os.path.join(output_dir, 'popular_articles.jsonl')),
        )

    def load_json(self, answer_dir, output_dir, file_name):
        self.path = os.path.join(output_dir, file_name)
        answer = json.load(open(os.path.join(answer_dir, file_name)))
        output = json.load(open(self.path))
        return answer, output

    def get_dirname(self) -> str:
        return os.path.dirname(self.path)

    def get_filename(self) -> str:
        return os.path.basename(self.path)

    def check_impl(self, *args, **kwargs):
        raise NotImplementedError


class CrawlChecker(Checker):
    def calc_iou(self, answer, output):
        num_intersection = len(answer & output)
        iou = num_intersection / (len(answer) + len(output) - num_intersection)
        return iou

    def check_impl(self, answer_dir, output_dir, threshold=0.9):
        """Override"""
        (answer_articles,
         answer_popular_articles,
         output_articles,
         output_popular_articles,) = self.load_jsonl(answer_dir, output_dir)

        iou_all_article = self.calc_iou(answer_articles, output_articles)
        iou_all_popular = self.calc_iou(answer_popular_articles, output_popular_articles)
        iou = (iou_all_article + iou_all_popular) / 2

        if iou > threshold:
            return {
                "score": 24,
                "message": (
                    f"[PASSED] Since the overlap ratio({iou:.2f}) > "
                    f"threshold({threshold:.2f}). "
                    f"Overlap ratio of articles.jsonl: {iou_all_article:.2f}, "
                    f"Overlap ratio of popular_articles.jsonl: {iou_all_popular:.2f}."),
            }
        else:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Since the overlap ratio({iou:.2f}) <= "
                    f"threshold({threshold:.2f}). "
                    f"Overlap ratio of articles.jsonl: {iou_all_article:.2f}, "
                    f"Overlap ratio of popular_articles.jsonl: {iou_all_popular:.2f}."),
            }


class PushChecker(Checker):
    def calc_order_iou(self, answer, output):
        answer_order = set(combinations(answer, 2))
        output_order = set(combinations(output, 2))
        num_intersection = len(answer_order & output_order)
        iou = num_intersection / (len(answer_order) + len(output_order) - num_intersection)
        return iou

    def extract_user_ids(self, result):
        push = [item['user_id'] for item in result['push']['top10']]
        boo = [item['user_id'] for item in result['boo']['top10']]
        return push, boo

    def check_impl(self, answer_dir, output_dir, file_name, count_error_threshold=0.1, username_threshold=0.8):
        answer, output = self.load_json(answer_dir, output_dir, file_name)
        answer_push, answer_boo = self.extract_user_ids(answer)
        output_push, output_boo = self.extract_user_ids(output)

        push_iou = self.calc_order_iou(answer_push, output_push)
        boo_iou = self.calc_order_iou(answer_boo, output_boo)
        iou = (push_iou + boo_iou) / 2

        push_error = abs(output['push']['total'] - answer['push']['total']) / answer['push']['total']
        boo_error = abs(output['boo']['total'] - answer['boo']['total']) / answer['boo']['total']
        error = (push_error + boo_error) / 2

        if iou > username_threshold and error < count_error_threshold:
            return {
                "score": 7,
                "message": (
                    f"[PASSED] Since the username overlap ratio({iou:.2f}) > "
                    f"threshold({username_threshold:.2f}) and the averaged count "
                    f"error({error:.2f}) < "
                    f"count_error_threshold({count_error_threshold:.2f}). "
                    f"Username overlap ratio of push: {push_iou:.2f}. "
                    f"Username overlap ratio of boo: {boo_iou:.2f}. "
                    f"Error of push count: {push_error:.2f}. "
                    f"Error of boo count: {boo_error:.2f}."),
            }
        else:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Since the username overlap ratio({iou:.2f}) <= "
                    f"threshold({username_threshold:.2f}) or the averaged count "
                    f"error({error:.2f}) >= "
                    f"count_error_threshold({count_error_threshold:.2f}). "
                    f"Username overlap ratio of push: {push_iou:.2f}. "
                    f"Username overlap ratio of boo: {boo_iou:.2f}. "
                    f"Error of push count: {push_error:.2f}. "
                    f"Error of boo count: {boo_error:.2f}."),
            }


class PopularChecker(Checker):
    def check_impl(self, answer_dir, output_dir, file_name, threshold=0.9):
        answer, output = self.load_json(answer_dir, output_dir, file_name)
        answer = set(answer['image_urls'])
        output = set(output['image_urls'])
        num_intersection = len(answer & output)
        iou = num_intersection / (len(answer) + len(output) - num_intersection)

        if iou > threshold:
            return {
                "score": 7,
                "message": (
                    f"[PASSED] Since url overlap ratio({iou:.2f}) > threshold("
                    f"{threshold:.2f})"),
            }
        else:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Since url overlap ratio({iou:.2f}) <= threshold("
                    f"{threshold:.2f})"),
            }


class KeywordChecker(Checker):
    def check_impl(self, answer_dir, output_dir, file_name, threshold=0.9):
        answer, output = self.load_json(answer_dir, output_dir, file_name)
        answer = set(answer['image_urls'])
        output = set(output['image_urls'])
        num_intersection = len(answer & output)
        iou = num_intersection / (len(answer) + len(output) - num_intersection)

        if iou > threshold:
            return {
                "score": 8,
                "message": (
                    f"[PASSED] Since url overlap ratio({iou:.2f}) > threshold("
                    f"{threshold:.2f})"),
            }
        else:
            return {
                "score": 0,
                "message": (
                    f"[FAILED] Since url overlap ratio({iou:.2f}) <= threshold("
                    f"{threshold:.2f})"),
            }


def eval(answer_dir: str, output_dir: str):
    results = {
        'crawl': CrawlChecker().check(answer_dir, output_dir)
    }

    # push, popular, keyword
    tasks = {
        'push': PushChecker(),
        'popular': PopularChecker(),
        'keyword': KeywordChecker(),
    }
    for task_name, checker in tasks.items():
        answer_files = sorted(
            glob.glob(os.path.join(answer_dir, f"{task_name}_*.json")))
        for answer_file in answer_files:
            file_name = os.path.basename(answer_file)
            results[file_name] = checker.check(answer_dir, output_dir, file_name)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('answer_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    results = eval(args.answer_dir, args.output_dir)

    for task_name, info in results.items():
        print(f"[{task_name}]")
        print(f"score : {info['score']}")
        print(f"deduct: {info['message']}")
        print("")
