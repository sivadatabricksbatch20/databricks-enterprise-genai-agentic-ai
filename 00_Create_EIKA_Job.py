# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Create the EIKA jobs
# MAGIC
# MAGIC Open this notebook from the project's Git folder and run all cells. It creates two serverless jobs.
# MAGIC Run **EIKA - Stage A (RAG)** first. After it succeeds, run **EIKA - Stage B and C**
# MAGIC with the same profile, because Stage B uses the index built by Stage A.
# MAGIC
# MAGIC Use the learning profile for the sample-data lab. Leave the UC catalog on auto to
# MAGIC resolve it from the workspace, or enter a catalog where you can create a schema.
# MAGIC Add a unique job name suffix if students share a workspace.
# MAGIC Running this notebook again from the same folder updates the two jobs.

# COMMAND ----------

dbutils.widgets.dropdown("profile", "learning", ["learning", "enterprise"], "Profile")
dbutils.widgets.dropdown("debug", "yes", ["no", "yes"], "Show configuration trace in tasks")
dbutils.widgets.text("catalog", "auto", "UC catalog for Stage C")
dbutils.widgets.text("schema", "eika", "UC schema for Stage C")
dbutils.widgets.text("name_suffix", "", "Job name suffix (optional, e.g. your initials)")
dbutils.widgets.dropdown("run_now", "no", ["no", "yes"], "Start Stage A after setup")

# COMMAND ----------

import json
import sys

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
here = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
ROOT = here.rsplit("/", 1)[0]
NB = f"{ROOT}/notebooks"
PROFILE = dbutils.widgets.get("profile")
DEBUG = dbutils.widgets.get("debug")
CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
SUFFIX = dbutils.widgets.get("name_suffix").strip()
RUN_NOW = dbutils.widgets.get("run_now") == "yes"


w.workspace.get_status(NB)
workspace_root = ROOT if ROOT.startswith("/Workspace/") else f"/Workspace{ROOT}"
sys.path.insert(0, workspace_root)
from src.utils.enterprise_setup import _workspace_catalog

CATALOG = _workspace_catalog(spark, CATALOG)
print("Project folder:", ROOT)
print("Stage C catalog and schema:", f"{CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define the tasks
# MAGIC Each task waits for the one before it, because Stage B reuses Stage A's index and Stage C reuses both.

# COMMAND ----------

uc = {"catalog": CATALOG, "schema": SCHEMA}

stage_a = [
    ("a1_architecture_setup", "stage_a_rag/01_Architecture_and_Setup", {}),
    ("a2_document_ingestion", "stage_a_rag/02_Document_Ingestion", {}),
    ("a3_text_chunking", "stage_a_rag/03_Text_Chunking", {}),
    ("a4_embedding_generation", "stage_a_rag/04_Embedding_Generation", {}),
    ("a5_vector_search", "stage_a_rag/05_Vector_Search", {}),
    ("a6_rag_pipeline", "stage_a_rag/06_RAG_Pipeline", {}),
    ("a7_rag_evaluation", "stage_a_rag/07_RAG_Evaluation", {}),
]

stage_bc = [
    ("b1_agent_architecture", "stage_b_agentic_ai/01_Agentic_AI_Architecture", {}),
    ("b2_tool_calling", "stage_b_agentic_ai/02_Tool_Calling", {}),
    ("b3_monitoring_agent", "stage_b_agentic_ai/03_Monitoring_Agent", {}),
    ("b4_knowledge_agent", "stage_b_agentic_ai/04_Knowledge_Agent", {}),
    ("b5_data_quality_agent", "stage_b_agentic_ai/05_Data_Quality_Agent", {}),
    ("b6_agent_orchestration", "stage_b_agentic_ai/06_Agent_Orchestration", {}),
    ("b7_agent_evaluation", "stage_b_agentic_ai/07_Agent_Evaluation", {}),
    ("c1_responses_agent", "stage_c_production/01_Serve_Agent_with_ResponsesAgent", dict(uc, deploy_endpoint="no", cleanup="no")),
    ("c2_vector_search_mcp", "stage_c_production/02_Vector_Search_and_MCP_Retrieval", dict(uc, create_resources="no", cleanup="no")),
    ("c3_evaluation_gates", "stage_c_production/03_Evaluation_Gates_and_Promotion", {"promote": "no"}),
    ("c4_databricks_app", "stage_c_production/04_Deploy_as_Databricks_App", {"deploy": "no", "teardown": "none"}),
]

def chain(items):
    """No cluster on a task means serverless. Each task depends on the previous one."""
    tasks, prev = [], None
    for key, path, params in items:
        task = {"task_key": key, "timeout_seconds": 3600,
                "notebook_task": {"notebook_path": f"{NB}/{path}", "source": "WORKSPACE", "base_parameters": params}}
        if prev:
            task["depends_on"] = [{"task_key": prev}]
        tasks.append(task)
        prev = key
    return tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update both jobs
# MAGIC Job creation leaves the app deployment exercise off. The hosted app configuration
# MAGIC in this repository points to the instructor's index and must be adapted before
# MAGIC a student deploys an app in a different workspace.

# COMMAND ----------

def save_job(name, description, tasks):
    name += f" - {SUFFIX}" if SUFFIX else ""
    settings = {
        "name": name,
        "description": description,
        "tags": {"project": "eika", "course": "swsacademy"},
        "max_concurrent_runs": 1,
        "parameters": [
            {"name": "profile", "default": PROFILE},
            {"name": "debug", "default": DEBUG},
        ],
        "tasks": tasks,
    }
    jobs = list(w.jobs.list(name=name))
    if len(jobs) > 1:
        raise ValueError(f"More than one job is named {name!r}. Set a unique job name suffix.")
    if jobs:
        job_id = jobs[0].job_id
        existing = w.jobs.get(job_id).settings
        first_task = next((task for task in existing.tasks or [] if not task.depends_on), None)
        current_path = (first_task.notebook_task.notebook_path or ""
                        if first_task and first_task.notebook_task else "")
        expected_path = tasks[0]["notebook_task"]["notebook_path"]
        if current_path.removeprefix("/Workspace") != expected_path.removeprefix("/Workspace"):
            raise ValueError(f"{name!r} belongs to another project folder. Set a unique job name suffix.")
        w.api_client.do("POST", "/api/2.2/jobs/reset", body={"job_id": job_id, "new_settings": settings})
        print(f"Updated {name}: {w.config.host}/jobs/{job_id}")
    else:
        job_id = w.api_client.do("POST", "/api/2.2/jobs/create", body=settings)["job_id"]
        print(f"Created {name}: {w.config.host}/jobs/{job_id}")
    return job_id


stage_a_job_id = save_job("EIKA - Stage A (RAG)", "Build and evaluate the RAG pipeline.", chain(stage_a))
stage_bc_job_id = save_job("EIKA - Stage B and C", "Run after Stage A succeeds with the same profile.", chain(stage_bc))

if RUN_NOW:
    run_id = w.api_client.do("POST", "/api/2.2/jobs/run-now", body={"job_id": stage_a_job_id})["run_id"]
    print(f"Started Stage A: {w.config.host}/jobs/{stage_a_job_id}/runs/{run_id}")

print("Run Stage B and C after Stage A succeeds.")
dbutils.notebook.exit(json.dumps({"stage_a_job_id": stage_a_job_id, "stage_bc_job_id": stage_bc_job_id}))
