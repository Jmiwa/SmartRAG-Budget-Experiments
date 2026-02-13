
# SmartRAG-Budget-Experiments

This repository provides the experimental structure and evaluation code for the SmartRAG-Budget framework, which generates household financial improvement advice using LLMs and Retrieval-Augmented Generation (RAG).

Due to copyright restrictions, the original consultation articles cannot be redistributed.
Instead, this repository includes synthetic sample data that preserves the structure used in the experiments.

---

## Dataset

In the original experiments, publicly available household consultation articles were collected and structured into JSON format, including:

- Consultation text
- Household financial data
- Expert advice
- (Optional) Reduction rate annotations

Because the original articles cannot be redistributed, this repository provides:

- Synthetic index samples for Vector DB construction
- Synthetic user input samples
- Sample output files
- The original prompt templates
- Evaluation and index-building scripts

The provided samples are anonymized and modified versions designed to preserve structure only.
They do not match the original articles.

---

## Directory Structure

```

data/
index_samples/         # Sample data for Vector DB indexing
with_reduction/      # Includes reduction rate annotations
no_reduction/        # Without reduction rate annotations
user_input_samples/    # Example user input JSON files

prompts/                 # Prompt templates used in experiments

scripts/
build_index/           # Pinecone index construction scripts
evaluation/            # LLM and RAG evaluation scripts

results/
sample_outputs/        # Example generated outputs

```

---

## Example Execution

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

### LLM-only

```

python scripts/evaluation/llm_only_eval.py

```

### RAG (with reduction rates)

```

python scripts/evaluation/rag_eval.py

```

### RAG (without reduction rates)

```

python scripts/evaluation/rag_eval_no_reduction.py

```

---

## Requirements

Execution requires:

- Gemini API key
- OpenAI API key (for embeddings)
- Pinecone API key

Set them in a `.env` file.

---

## Notes

- Outputs may vary depending on model versions and retrieval results.
- This repository focuses on releasing the experimental framework and structure.
- Full reproducibility is not possible because the original dataset cannot be redistributed.

---

## License

MIT License