import json
import os
import sys
import traceback as tr
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from huggingface_hub.utils import HFValidationError

from dialog_processor import BatchDialogProcessor, DialogProcessor

sys.path.append(str(Path(__file__)))
os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class InvalidAnswerType(Exception):
    """
    Raised when the type of the answer is wrong.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, func_name: str, ret_type: str, message: Optional[str] = None):
        self.message = (
            message
            if message is not None
            else f"The answer returned from `{func_name}` function should be of type `{ret_type}`. Please, check the returned value type."
        )
        super().__init__(self.message)


def check_errors(errors: Union[list, str, dict], init_path: str):
    """
    Check whether any errors during evaluation occurred.
    :param errors: a dict/list/str of error logs;
    :param init_path: a path to save log;
    :return: None
    """
    if errors:
        if not os.path.exists(init_path):
            os.makedirs(init_path)
        with open(f"{init_path}/error.json", "w") as fp:
            json.dump(errors, fp, ensure_ascii=False)
        exit("Error is written to error.json")


try:
    from submit import SubmitModelWithMemory
except ImportError as e:
    exception_det = f"Please, provide the ModelWithMemory implementation that can be imported from the `submit`, and check the availability of the imported modules and libraries.\n{e}"
    print("ImportError: " + exception_det)
    errors = {"exception": exception_det}
    check_errors(errors, "./")


def generate_answers_with_memory(dataset_path: str, output_dir: str, model_path: str):
    processor = BatchDialogProcessor(
        dialog_processor=DialogProcessor(
            model_with_memory=SubmitModelWithMemory(model_path)
        )
    )
    processor.process(dataset_path, output_dir)


def launch_inference(dataset_path: str, output_path: str, model_path: str):
    disable_torch_init()

    os.makedirs(output_path, exist_ok=True)
    start_time = datetime.now()

    errors = {}
    try:
        generate_answers_with_memory(dataset_path, output_path, model_path)

    except FileNotFoundError as e:
        error = tr.TracebackException(
            exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e
        ).stack[-1]
        exception_det = "{} in {} row".format(e, error.lineno)
        print("FileNotFoundError: " + exception_det)
        errors["exception"] = (
            "FileNotFoundError: Some problem with file's path occurred.\nCheck the correct path for "
            "the models' weights. "
        )
        return errors
    except HFValidationError as e:
        error = tr.TracebackException(
            exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e
        ).stack[-1]
        exception_det = "{} in {} row".format(e, error.lineno)
        print("HFValidationError: " + exception_det)
        errors["exception"] = (
            "HFValidationError: Some problem with downloading hf checkpoints.\nPlease, remember that "
            "there is no Internet access from the job and all the files (including models' weights) "
            "should be either uploaded within the docker image, or the submission zip. "
        )
        return errors
    except IndexError as e:
        print(f"IndexError: {str(e)}")
        errors["exception"] = "IndexError: Check the correctness of the indices."
        return errors
    except TypeError as e:
        print(f"TypeError: {str(e)}")
        errors["exception"] = "TypeError: Check the correctness of the data types."
        return errors
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        errors["exception"] = (
            "ValueError: This error is probably (but not necessarily) a result of wrong returned "
            "values, please, check whether all values are returned correctly. "
        )
        return errors
    except InvalidAnswerType as e:
        exception_det = e.message
        print("ValueError: " + exception_det)
        errors["exception"] = f"InvalidAnswerTypeError: {exception_det}"
        return errors
    except ImportError as e:
        print(f"ImportError: {str(e)}")
        errors["exception"] = (
            "ImportError: Check the availability of the imported modules and libraries."
        )
        return errors
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {str(e)}")
        errors["exception"] = (
            "ImportError: Check the availability of the imported modules and libraries."
        )
        return errors
    except Exception as e:
        error = tr.TracebackException(
            exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e
        ).stack[-1]
        exception_det = "{} in {} row".format(e, error.lineno)
        print(f"Exception: {exception_det}")
        errors["exception"] = (
            "Exception: Some error occurred during the run, please refer to the logs."
        )
        return errors

    print(datetime.now() - start_time)
    return None


def launch_inference_and_check_errors(
    dataset_path: str, output_path: str, model_path: str
):
    errors = launch_inference(dataset_path, output_path, model_path)
    check_errors(errors, output_path)


if __name__ == "__main__":
    launch_inference_and_check_errors(
        dataset_path="../data/leaderboard_data.jsonl",
        output_path="../output",
        model_path="/app/models/GigaChat-20B-A3B-instruct-v1.5-bf16"
    )
