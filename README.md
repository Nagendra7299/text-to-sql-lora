# Text-to-SQL — Mistral-7B Fine-Tuned with QLoRA

> Fine-tune a 7-billion-parameter LLM to turn plain-English questions into SQL queries — trained on a single free GPU using QLoRA.

**[Adapter on HF Hub](https://huggingface.co/Nagendra729/mistral-7b-text-to-sql-lora)** · **[Training Notebook](./train_mistral_sql_qlora.ipynb)** · **[Try it in Colab](#-demo)**

---

## 🎬 Demo

<!-- Record the snippet below running and drop the GIF here:
     1. Run the "Try it yourself" cell in Colab
     2. Screen-record it generating SQL for 2-3 questions
     3. Convert to GIF (e.g. ezgif.com) and save as assets/demo.gif
     4. Uncomment the line below -->
<!-- ![Text-to-SQL demo](assets/demo.gif) -->

> **GIF coming soon** — record the model converting English → SQL in real time (see snippet below).

### Try it yourself (free, no setup)

Paste into a [Google Colab](https://colab.research.google.com) cell (Runtime → T4 GPU) and run — the adapter is public, no token needed:

```python
!pip install -q unsloth
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    "Nagendra729/mistral-7b-text-to-sql-lora",
    max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

def ask(schema, question):
    p = (f"You are a precise text-to-SQL engine. Given a database schema and a "
         f"question, output ONLY the SQL query.\n\n### Schema:\n{schema}\n\n"
         f"### Question:\n{question}\n\n### SQL:\n")
    out = model.generate(**tok(p, return_tensors="pt").to("cuda"),
                         max_new_tokens=128, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True).split("### SQL:")[-1].strip())

ask("CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT)",
    "average salary in the engineering department")
ask("CREATE TABLE orders (id INT, customer TEXT, amount INT, status TEXT)",
    "list customers with pending orders over 500")
```

Expected output:
```sql
SELECT AVG(salary) FROM employees WHERE department = 'engineering'
SELECT customer FROM orders WHERE status = 'pending' AND amount > 500
```

> A base-vs-fine-tuned comparison (exact-match score + side-by-side SQL) is generated in the [training notebook](./train_mistral_sql_qlora.ipynb), section 7.

---

## What This Project Shows

Most AI portfolios only *call* an API (OpenAI, Groq). This one works **below the API layer** — it takes an open-source base model and actually changes its weights for a new skill.

The result: ask a question in English, provide your table structure, and the model writes the SQL query.

**Example**

> **Schema:** `CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT)`
> **Question:** "What is the average salary in the engineering department?"
> **Model output:** `SELECT AVG(salary) FROM employees WHERE department = 'engineering'`

---

## For Non-Technical Readers

Imagine a brilliant new hire who knows language but has never seen your company's database. **Fine-tuning** is on-the-job training: you show the model thousands of examples of "question → correct SQL" until it learns the pattern — without forgetting everything else it knows.

The clever part is **doing it cheaply**. A 7-billion-parameter model normally needs expensive data-center GPUs to train. Two techniques fix that:

- **LoRA** — instead of re-training all 7 billion knobs, we add a tiny set of new knobs (about 0.5% of the total) and only train those. 200× fewer things to adjust.
- **QLoRA** — we also *compress* the frozen original model from 16-bit to 4-bit numbers, cutting memory by 4×.

Together they let this train on a **single free Google Colab GPU** in under an hour — something that used to require a cluster.

---

## Technical Summary

| | |
|---|---|
| **Base model** | `mistralai/Mistral-7B-Instruct-v0.3` (via Unsloth 4-bit) |
| **Method** | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| **Dataset** | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78k (question, schema, SQL) triples |
| **LoRA rank** | 16 (alpha 16, dropout 0) |
| **Trainable params** | ~42M of 7.25B (~0.6%) |
| **Hardware** | 1× free Colab T4 (16 GB) |
| **Frameworks** | Unsloth · PEFT · TRL · bitsandbytes · Transformers |
| **Outputs pushed to Hub** | LoRA adapter + merged 16-bit + GGUF Q4_K_M |

### Why QLoRA over full fine-tuning
- Full fine-tune of 7B in fp16 ≈ 80+ GB VRAM (needs A100). QLoRA ≈ 6–8 GB → fits a free T4.
- Adapter is ~80 MB vs ~14 GB for full weights → cheap to store, version, and swap.
- Base model stays frozen → no catastrophic forgetting of general ability.

---

## How to Reproduce

### 1. Train (free, on Colab)
Open `train_mistral_sql_qlora.ipynb` in Google Colab → set runtime to **T4 GPU** → run all cells.
You'll be prompted for your **Hugging Face token** (write access) to push the result.

The notebook:
1. Loads Mistral-7B in 4-bit (Unsloth)
2. Formats `sql-create-context` into instruction prompts
3. Attaches LoRA adapters, trains with `SFTTrainer`
4. Evaluates **base vs. fine-tuned** on held-out questions (side-by-side SQL)
5. Pushes adapter + GGUF to your HF Hub

### 2. Demo
The `space/` folder is a Gradio app that loads the GGUF on CPU and serves the model. See [`space/README.md`](./space/README.md).

---

## Results

The fine-tuned model produces clean, schema-aware SQL where the base model often rambles, adds prose, or hallucinates columns. Quantitative exact-match and qualitative examples are generated in the notebook's eval section.

*(Run the notebook to populate your own metrics table here.)*

---

## Repo Structure

```
text-to-sql-lora/
├── train_mistral_sql_qlora.ipynb   # Colab training notebook (core deliverable)
├── README.md                       # This file
├── requirements.txt                # Local/reference deps
└── space/                          # Gradio inference demo (HF Space)
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
```

---

## Author

Built by **Nagendra Chowdary** · [GitHub](https://github.com/Nagendra7299) · [Portfolio](https://portfolio-one-chi-cvtu2ez2sd.vercel.app)
