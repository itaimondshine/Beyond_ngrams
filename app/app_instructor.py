import os
import pathlib
import random
import re
import sys
import traceback
import uuid
from datetime import datetime, timedelta

import firebase_admin
import google.generativeai as genai
import pandas as pd
from firebase_admin import firestore
from flask import Flask, render_template, request, jsonify
from flask import redirect, url_for
from flask_session import Session
from google.cloud.firestore_v1.base_query import FieldFilter, And

import tasks

# from utils import load_xlsum_data

N_TASKS_PER_USER = int(os.environ.get("N_TASKS", 5))
SANDBOX_ON = False

LANGUAGE = os.environ.get("LANGUAGE")
DEFAULT_ARGS = os.environ.get("DEFAULT_ARGS", "1A")


# Initialize Firestore DB
cred = firebase_admin.credentials.Certificate("new_key.json")
default_app = firebase_admin.initialize_app(cred)
db = firestore.client()
instructions_ref = db.collection("instructions")
instructions_ref_sandbox = db.collection("instructions_sandbox")

instructions_ref_get = list(instructions_ref.get())
instructions_ref_sandbox_get = list(instructions_ref_sandbox.get())

# Create a collection for the language
summaries_ref = db.collection(f"{LANGUAGE}2")
info_ref = db.collection("info_experiment2")

logs = db.collection("logs")
global_variables_db = db.collection("global_variables")

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = uuid.uuid4().hex
app.app_context().push()
Session(app)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)


LANGUAGE_TO_LANGUAGE_CODE = {
    "arabic": "ar",
    "spanish": "es",
    "english": "en",
    "japanese": "ja",
    "chinese": "ch",
    "turkish": "tr",
    "indonesian": "in",
    "yoruba": "yo",
    "vietnamese": "vi",
    "ukrainian": "uk",
}


def log_info(content):
    j_req = {"content": content, "date": str(datetime.utcnow())}
    logs.document(str(random.randint(0, 100000))).set(j_req)
    print(content)


def update_global_variables(global_variables, session_id):
    global_variables_db.document(session_id).set(global_variables.to_dict())


def validate_answered_question(q_idx, user=DEFAULT_ARGS):
    query = summaries_ref.where(
        filter=And(
            [
                FieldFilter("work_id", "==", user),
                FieldFilter("n_sample", "==", q_idx),
            ]
        )
    )
    docs = query.stream()
    return True if len(list(docs)) >= 1 else False


# def get_sample(workerId=DEFAULT_ARGS):
#     language_data_path = pathlib.Path(__file__).parent / "app_data" / "corrupted"/ f"{LANGUAGE}.csv"
#     language_data_df = pd.read_csv(language_data_path)
#     # Get full indices
#     full_indices = pd.read_csv(str(language_data_path))['q_idx'].tolist()
#     summary_doc = dict(info_ref.document(LANGUAGE).get().to_dict())
#     if not summary_doc:
#         summary_doc.set({str(i): 0 for i in full_indices})
#     # filtered_keys = [k for k, v in summary_doc.items() if v < 2]
#     # Get random choice
#     selected_key = random.randint(1,50)
#     # while validate_answered_question(workerId, selected_key):
#     #     selected_key = random.choice(filtered_keys)
#     data = language_data_df[language_data_df['q_idx'] == int(selected_key)]
#     return data.iloc[0].to_dict()


def get_sample(workerId=DEFAULT_ARGS):
    language_data_path = (
        pathlib.Path(__file__).parent / "app_data" / "corrupted" / "c" / f"{LANGUAGE}.csv"
    )
    language_data_df = pd.read_csv(language_data_path)
    # Get full indices
    full_indices = set(
        pd.read_csv(str(language_data_path))["q_idx"].tolist()
    ).intersection(set(language_data_df["q_idx"]))
    summary_doc = dict(info_ref.document(LANGUAGE).get().to_dict())
    if not summary_doc:
        info_ref.document(LANGUAGE).set({str(i): 0 for i in full_indices})
        summary_doc = dict(info_ref.document(LANGUAGE).get().to_dict())
    filtered_keys = [k for k, v in summary_doc.items() if v < 3]
    # Get random choice
    selected_key = random.choice(filtered_keys)
    while validate_answered_question(int(selected_key), workerId):
        selected_key = random.choice(filtered_keys)
    selected_key = int(selected_key)
    data = language_data_df[language_data_df["q_idx"] == selected_key]
    return data.iloc[0].to_dict()


@app.route("/", methods=["GET", "POST"])
@app.route("/home/", methods=["GET", "POST"])
def home():
    assignmentId = request.args.get("assignmentId", str(datetime.now().hour))
    hitId = request.args.get("hitId", str(datetime.now().hour))
    turkSubmitTo = request.args.get("turkSubmitTo", "https://workersandbox.mturk.com")
    workerId = request.args.get("workerId", DEFAULT_ARGS)

    session_id = str(workerId + assignmentId + hitId)

    global_variables = init_global_variable(session_id)

    return description_task(
        global_variables, workerId, hitId, assignmentId, session_id, turkSubmitTo
    )


def init_global_variable(session_id):
    global instructions_ref_get
    global instructions_ref

    global_variables_specific = global_variables_db.document(session_id)
    doc = global_variables_specific.get()

    if doc.to_dict():
        global_variables = tasks.Tasks.from_dict(doc=doc.to_dict())
    else:
        global_variables = tasks.Tasks(sessions_id=session_id)
        global_variables_specific.set(global_variables.to_dict())
    return global_variables


def end_hit(global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId):
    global instructions_ref_get, instructions_ref

    address = turkSubmitTo + "/mturk/externalSubmit"
    fullUrl = address + "?assignmentId=" + assignmentId

    try:
        global_variables.finished = True
        global_variables.task_n = 0
        global_variables_db.document(session_id).set(global_variables.to_dict())
    except:
        write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId)

    instructions_ref_get = list(instructions_ref.get())

    return render_template("end.html", bar=100, fullUrl=fullUrl)


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
        "url": request.url,
    }

    if global_variables and global_variables.session_id:
        j_req["task_session_n"] = global_variables.task_n
        j_req["start_session"] = global_variables.start_session

    logs.document(id).set(j_req)


def write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId):
    _, _, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
    log_state = (
        "Error LOG"
        if "ASSIGNMENT_ID_NOT_AVAILABLE" not in request.url
        else "Warning LOG"
    )
    stack_trace = [log_state]

    for trace in trace_back:
        stack_trace.append(
            "File: %s, Line: %d, Func.Name: %s, Message: %s"
            % (trace[0], trace[1], trace[2], trace[3])
        )

    write_log(global_variables, workerId, hitId, session_id, assignmentId, stack_trace)


def handle_error(
    global_variables, error, workerId, hitId, session_id, assignmentId, turkSubmitTo
):
    write_error_to_log(global_variables, workerId, hitId, session_id, assignmentId)
    return end_hit(
        global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId
    )


def update_task_number(global_variables, session_id):
    global_variables.task_n += 1
    update_global_variables(global_variables, session_id)


def gemini_completion(prompt):
    # Define the endpoint URL
    genai.configure(api_key="AIzaSyBgja1_jiDdyVsIA0KBamDdgECkZ7r32y8")
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    return model.generate_content(prompt).text


def get_gpt_grade():
    text = request.form["summary"]
    gpt_question = request.form["gpt_question"]
    answer = request.form["answer"]
    prompt = (
        f"Text: '{text}' "
        f"\nQuestion: '{gpt_question}'"
        f"\nAnswer: {answer}"
        f"\nHow good is the given answer? 1 is for bad answer and 5 for very good answer"
    )
    gpt_answer = gemini_completion(prompt)
    return gpt_answer


def normalize_value(value: str) -> str:
    value = int(value)
    if value in (4,5):
        return str(value - 1)
    else:
        return str(value)

def post_descriptions(
    global_variables, data, workerId, hitId, session_id, assignmentId, turkSubmitTo
):
    try:
        id = workerId + hitId + str(global_variables.task_n)
        q_idx = data["q_idx"]
        gemini_pred = data["gemini_prediction"]
        summary = data["label"]
        gpt_pred = data["gpt_prediction"]

        original_gpt_grade = get_gpt_grade()

        match = re.search(r"\d+", original_gpt_grade)
        #
        gpt_grade = int(match.group()) if match else None

        j_req = {
            "hit_id": hitId,
            "work_id": workerId,
            "assignmentId": assignmentId,
            "summary": summary,
            "gpt_prediction": gpt_pred,
            "gemini_prediction": gemini_pred,
            "coherence_gpt": normalize_value(request.form["coherence_gpt"]),
            "consistence_gpt": normalize_value(request.form["consistence_gpt"]),
            # "fluency_gpt": normalize_value(request.form["fluency_gpt"]),
            # "relevance_gpt": normalize_value(request.form["relevance_gpt"]),
            "coherence_gemini": normalize_value(request.form["coherence_gemini"]),
            "consistence_gemini": normalize_value(request.form["consistence_gemini"]),
            # "fluency_gemini": normalize_value(request.form["fluency_gemini"]),
            # "relevance_gemini": normalize_value(request.form["relevance_gemini"]),
            "gpt_question": request.form["gpt_question"],
            "answer": request.form["answer"],
            "location": request.form["location"],
            "task": global_variables.task_n,
            "date_start": str(global_variables.start_session),
            "date_finish": str(datetime.utcnow()),
            "n_sample": q_idx,
            "gpt_grade": gpt_grade,
            "original_gpt_grade": original_gpt_grade,
            "key": id,
            "feedback": request.form['mismatch']
        }

        if "sandbox" in turkSubmitTo and SANDBOX_ON:
            instructions_ref_sandbox.document(id).set(j_req)
        else:
            summaries_ref.document(id).set(j_req)

        # update Info
        info_ref.document(LANGUAGE).update({str(q_idx): firestore.Increment(1)})

        # global_variables.finished = True
        update_global_variables(global_variables, session_id)

    except Exception as e:
        return f"An Error Occurred: {e}"

    update_task_number(global_variables, session_id)

    if global_variables.task_n > N_TASKS_PER_USER:
        return end_hit(
            global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId
        )
    else:
        return redirect(url_for("home"))


def get_gpt_question(text: str):
    prompt = (
        f"Ask a question on the following text in {LANGUAGE}: {text}."
        f"The Question should be one sentence maximum, and the answer should be in the text"
    )
    return gemini_completion(prompt)


def description_task(
    global_variables, workerId, hitId, assignmentId, session_id, turkSubmitTo
):
    try:
        data = get_sample(workerId)

        print("got data")

        all_data_path = (
                pathlib.Path(__file__).parent / "app_data" / "xlsum" / f"{LANGUAGE}.csv"
        )
        all_data = pd.read_csv(all_data_path)
        q_idx = data["q_idx"]
        gemini_pred = data["gemini_prediction"]
        summary = data["label"]
        article = all_data[all_data['q_idx'] == q_idx].iloc[0]['text']
        gpt_pred = data["gpt_prediction"]

        try:
            gpt_question = get_gpt_question(text=summary)
        except Exception:
            print("found error in gpt question")

        if request.method == "POST":
            post_descriptions(
                global_variables,
                data,
                workerId,
                hitId,
                session_id,
                assignmentId,
                turkSubmitTo,
            )

        global_variables.start_session = str(datetime.utcnow())
        update_global_variables(global_variables, session_id)

        if global_variables.task_n == N_TASKS_PER_USER:
            return end_hit(
                global_variables, session_id, turkSubmitTo, assignmentId, workerId, hitId
            )

        html_template = "task_template.html"

        article_url = (
            f"https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/{LANGUAGE}/test?row={q_idx}"
            if LANGUAGE != "hebrew"
            else f"https://huggingface.co/datasets/biunlp/HeSum/viewer/default/test?row={q_idx}"
        )

        return render_template(
            html_template,
            summary=article[:2500],
            gemini=gemini_pred,
            gpt=gpt_pred,
            n_sample=q_idx,
            gpt_question=gpt_question,
            article_url=article_url,
            language=LANGUAGE,
        )

    except Exception as e:
        return handle_error(
            global_variables, e, workerId, hitId, session_id, assignmentId, turkSubmitTo
        )


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html", exc=e)


port = int(os.environ.get("PORT", 5002))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)
