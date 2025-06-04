import flask
import sys
import random
from datetime import datetime
import traceback
from datasets import load_dataset
import numpy as np


def load_xlsum_data(lang, split, limit):
    """Loads the xlsum dataset"""
    dataset = load_dataset("csebuetnlp/xlsum", lang)[split]
    return dataset.select(np.arange(limit))


def write_log(global_variables, workerId, hitId, session_id, assignmentId, content):
    id = str(random.randint(0, 1000000))

    j_req = {
        "date": str(datetime.utcnow()),
        "hit_id": hitId,
        "work_id": workerId,
        "session_id": session_id,
        "assignmentId": assignmentId,
        "key": id,
        "content": content,
        "url": flask.request.url,
    }

    validation_log = ""
    if global_variables and global_variables.session_id:
        if global_variables.task_n - 1 < len(global_variables.validations):
            validation_log = global_variables.validations[
                global_variables.task_n - 1
            ].rvs_path
        j_req["task_session_n"] = global_variables.task_n
        j_req["sample_session_n"] = global_variables.write.writing_task_n
        j_req["start_session"] = global_variables.start_session

    j_req["validation_session"] = validation_log
    # save to database
    logs.document(id).set(j_req)


def write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId):
    # Get current system exception

    _, _, ex_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    log_state = "Error LOG"
    if "ASSIGNMENT_ID_NOT_AVAILABLE" in flask.request.url:
        log_state = "Warning LOG"
    stack_trace = [log_state]

    for trace in trace_back:
        stack_trace.append(
            "File : %s , Line : %d, Func.Name : %s, Message : %s"
            % (trace[0], trace[1], trace[2], trace[3])
        )

    write_log(global_variables, workerId, hitId, session_id, assignmentId, stack_trace)
    print(stack_trace)
