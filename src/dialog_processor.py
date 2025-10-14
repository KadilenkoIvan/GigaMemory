from dataclasses import asdict
from datetime import datetime
from typing import List


from tqdm.auto import tqdm
import sys
import time

from models import Dialog, DialogResponse
from submit_interface import ModelWithMemory
from utils.io_utils import read, save_result


class DialogProcessor:

    def __init__(self, model_with_memory: ModelWithMemory):
        self.model_with_memory = model_with_memory

    def process(self, dialog: Dialog) -> DialogResponse:
        dialog_messages = dialog.get_messages()

        start_time = datetime.now()
        message_batch = []
        for i, msg in enumerate(dialog_messages):
            message_batch.append(msg)
            if i % 2 == 1:
                self.model_with_memory.write_to_memory(message_batch, dialog.id)
                message_batch = []

        answer = self.model_with_memory.answer_to_question(dialog.id, dialog.question)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        self.model_with_memory.clear_memory(dialog.id)

        return DialogResponse(
            id=dialog.id,
            question=dialog.question,
            pred_answer=answer,
            success=True,
            answer_time=processing_time,
        )


class BatchDialogProcessor:

    def __init__(self, dialog_processor: DialogProcessor):
        self.dialog_processor = dialog_processor

    def process_batch_from_file(self, input_file: str):
        start_time = datetime.now()
        requests = read(input_file)
        responses = self.process_batch(requests)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        result = {
            "success": True,
            "total_dialogs": len(requests),
            "processing_time": processing_time,
            "responses": [asdict(r) for r in responses],
        }
        return result

    def process_batch(self, requests: List[Dialog]) -> List[DialogResponse]:
        result = []
        total = len(requests)
        for i, r in enumerate(tqdm(requests, total=total)):
            result.append(self.dialog_processor.process(r))
            print(f"Processed {i+1}/{total} items")
        return result

    def process(self, input_file: str, output_dir: str):
        result = self.process_batch_from_file(input_file)
        save_result(result, output_dir)
