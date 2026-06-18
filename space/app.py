import os
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Repo + file produced by the training notebook.
GGUF_REPO = os.environ.get("GGUF_REPO", "Nagendra729/mistral-7b-text-to-sql-gguf")
GGUF_FILE = os.environ.get("GGUF_FILE", "mistral-7b-text-to-sql.Q4_K_M.gguf")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional: set as a Space secret if the model is private

print(f"Loading {GGUF_REPO}/{GGUF_FILE} ...")
model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE, token=HF_TOKEN)
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
print("Model loaded.")

PROMPT = """You are a precise text-to-SQL engine. Given a database schema and a question, output ONLY the SQL query.

### Schema:
{schema}

### Question:
{question}

### SQL:
"""


def to_sql(schema, question):
    if not schema.strip() or not question.strip():
        return "⚠️ Provide both a table schema and a question."
    out = llm(
        PROMPT.format(schema=schema.strip(), question=question.strip()),
        max_tokens=160,
        temperature=0.0,
        stop=["\n\n", "###", "</s>"],
    )
    sql = out["choices"][0]["text"].strip()
    return sql or "(model returned empty output — try rephrasing)"


EXAMPLES = [
    [
        "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT)",
        "What is the average salary in the engineering department?",
    ],
    [
        "CREATE TABLE orders (order_id INT, customer TEXT, amount INT, status TEXT)",
        "List all customers with orders over 500 that are still pending.",
    ],
    [
        "CREATE TABLE matches (id INT, team TEXT, goals INT, season INT)",
        "Which team scored the most goals in season 2023?",
    ],
]

with gr.Blocks(title="Text-to-SQL · Mistral-7B QLoRA", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Text-to-SQL — Mistral-7B fine-tuned with QLoRA
Give a table schema and a plain-English question — the fine-tuned model writes the SQL.
Running the 4-bit GGUF on a **free CPU Space**, so generation takes a few seconds.
        """
    )
    with gr.Row():
        with gr.Column():
            schema = gr.Textbox(
                label="Database schema (CREATE TABLE ...)",
                lines=4,
                placeholder="CREATE TABLE users (id INT, name TEXT, age INT)",
            )
            question = gr.Textbox(
                label="Question (plain English)",
                lines=2,
                placeholder="How many users are older than 30?",
            )
            btn = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            output = gr.Code(label="Generated SQL", language="sql")

    btn.click(fn=to_sql, inputs=[schema, question], outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=[schema, question])

    gr.Markdown(
        f"""
---
**Base:** Mistral-7B-Instruct · **Method:** QLoRA (4-bit + LoRA) · **Trained on:** free Colab T4
· **Model:** [`{GGUF_REPO}`](https://huggingface.co/{GGUF_REPO})
        """
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
