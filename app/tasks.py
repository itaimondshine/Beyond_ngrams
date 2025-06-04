from datetime import datetime
import numpy

class Tasks:
    def __init__(self, sessions_id=None, task_n=0):

        self.session_id = sessions_id
        self.start_session = str(datetime.utcnow())
        self.task_n = 0

        self.finished = False

    def to_dict(self):
        task_dict = {}
        task_dict["session_id"] = self.session_id
        task_dict["start_session"] = self.start_session
        task_dict["task_n"] = self.task_n
        task_dict["finished"] = self.finished

        return task_dict

    @staticmethod
    def from_dict(doc):
        task = Tasks(sessions_id=doc["session_id"])
        task.start_session = doc["start_session"]
        task.task_n = doc["task_n"]
        task.finished = doc["finished"]
        return task
