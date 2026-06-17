# Evaluating PluRule

Evaluate vision-language and API models on the PluRule benchmark using
clustered datasets with multimodal Reddit discussion context.

## Features

- **Multiple Model Support**: Qwen3-VL vLLM models and OpenAI API models configured in `eval/config.py`
- **Configurable Contexts**: Control whether prompts include submission text, media, discussion context, and user labels
- **Prompt Variants**: Evaluate baseline and optional prompt-injection variants
- **Two-Stage Evaluation**: Generate reasoning first, then extract clean answers
- **Comprehensive Metrics**: Overall, per-rule-cluster, and per-subreddit-cluster accuracy

## Directory Structure

```
PluRule/
├── eval/
│   ├── config.py           # Model, context, and phrase configurations
│   ├── helpers.py          # Utility functions for data loading, prompting, evaluation
│   ├── evaluate.py         # Main evaluation script
│   └── build_rag_retrieval.py
├── output/
│   └── eval/               # Evaluation results (reasoning + performance JSONs)
│       ├── rag/            # Precomputed target-comment similarity artifacts
│       └── {model}/
│           └── {split}/
│               └── {context}/
│                   └── {phrase}_{mode}/
│                       ├── reasoning_TIMESTAMP.json
│                       ├── performance_TIMESTAMP.json
│                       └── rag-kK-filter-balance-src-SPLIT-art-HASH/
│                           ├── reasoning_TIMESTAMP.json
│                           └── performance_TIMESTAMP.json
└── logs/
    └── eval/               # Evaluation logs
        └── {model}/
            └── {split}/
                └── {context}/
                    └── {phrase}_{mode}/
                        ├── evaluation_TIMESTAMP.log
                        └── rag-kK-filter-balance-src-SPLIT-art-HASH/
                            └── evaluation_TIMESTAMP.log
```

## Installation

Use the bundled evaluation environment:

```bash
conda env create -f environment-eval.yml
conda activate plurule-eval
```

## Usage

### Media paths

For `submission-media` contexts, `submission.media_files` should contain paths
relative to the media root, using this layout:

```text
<subreddit>/<image-file>
```

Eval resolves those paths under `data/media` by default. To use another media
directory:

```bash
export PLURULE_EVAL_MEDIA_DIR=/path/to/media
```

### Basic Usage

```bash
# Evaluate Qwen3-VL-8B-Instruct on the test set with submission + discussion context
python eval/evaluate.py \
    --model qwen3-vl-8b-instruct \
    --split test \
    --context submission-discussion \
    --phrase cot \
    --mode prefill

# Debug mode (only 5 thread pairs)
python eval/evaluate.py \
    --model qwen3-vl-8b-instruct \
    --split test \
    --context none \
    --phrase baseline \
    --mode prefill \
    --debug
```

### RAG Few-Shot Retrieval

Build the dense target-comment similarity artifact once:

```bash
python eval/build_rag_retrieval.py \
    --query-split test \
    --candidate-split train \
    --cuda 0
```

Then enable retrieved few-shot examples during eval:

```bash
python eval/evaluate.py \
    --model qwen3-vl-8b-instruct \
    --split test \
    --context submission-discussion \
    --phrase grounded_context \
    --mode prefill \
    --rag-k 4 \
    --rag-filter rule-cluster \
    --rag-balance mixed
```

Few-shot examples use the same prompt context as the target eval case. Retrieval
is scored only on target-comment text; examples include their question, MCQ
options, and a generic labeled answer.

### Arguments

- `--model, -m`: Model to evaluate
  - vLLM: `qwen3-vl-4b-instruct`, `qwen3-vl-8b-instruct`, `qwen3-vl-30b-instruct`, `qwen3-vl-4b-thinking`, `qwen3-vl-8b-thinking`, `qwen3-vl-30b-thinking`
  - OpenAI API path: `gpt-4o`, `gpt5.2-low`, `gpt5.2-high`
  - `claude-sonnet-4` is present in `API_MODELS`, but the current API runner in `helpers.evaluate_two_stage_api()` is wired to OpenAI Flex

- `--split, -s`: Dataset split (`train`, `val`, `test`, `delta`)

- `--context, -c`: Dash-separated context flags. Subreddit metadata, rules, and the target comment are always included.
  - `none`: no optional context
  - `submission`: include submission text
  - `submission-media`: include submission text and media
  - `submission-discussion`: include submission text and full discussion thread
  - `submission-discussion-user`: include submission, discussion, and anonymized user labels
  - `submission-media-discussion-user`: include all optional context

- `--phrase, -p`: Prompting phrase
  - `baseline`: No additional phrase
  - `cot`: "Let's think step by step"
  - `analyze`: "Let's carefully analyze this content"
  - `artifacts`: "Let's look for rule violations"
  - `rules`: "Let's compare this against the subreddit rules"
  - `grounded_choice`: "Let's first examine the context around the target comment, then compare it with the subreddit rules and listed answer options"
  - `grounded_target`: "Let's first examine the subreddit, rules, submission, and discussion only insofar as they clarify the target comment, then choose from the listed options"
  - `grounded_context`: "Let's first examine the context that grounds the target comment: the subreddit, its rules, the submission, and the discussion, before choosing from the listed options"

- `--mode`: Phrase injection mode
  - `prefill`: Append phrase after chat template (default)
  - `prompt`: Append a prompt-mode rewrite to the question text

- `--cuda`: CUDA device IDs (default: `"1"`)
- `--debug`: Run with only 5 thread pairs for testing
- `--override`: Overwrite existing results
- `--max-response-tokens`: Maximum generation length for Stage 1 responses (default: 2048)
- `--rag-k`: Number of retrieved few-shot examples per target thread. `0` disables RAG.
- `--rag-retrieval-path`: Optional path to the `.pt` retrieval artifact. Defaults to `output/eval/rag/{split}_to_{rag-source-split}_target_comment_similarity.pt`.
- `--rag-filter`: Retrieval filter: `none`, `subreddit`, `subreddit-cluster`, or `rule-cluster`.
- `--rag-balance`: `mixed` balances violating/compliant examples before filling from nearest neighbors; `top` uses nearest neighbors only.
- `--rag-source-split`: Candidate split used by the retrieval artifact. Defaults to `train`.

## Output Format

### Reasoning JSON

Each thread pair produces two predictions (violating and compliant):

```json
{
  "mod_comment_id": "fdzc60l",
  "subreddit": "excel",
  "submission_id": "en8cqn",

  "violating": {
    "reasoning_response": "Let me analyze this comment...",
    "clean_answer_response": "(b)",
    "extracted_prediction": "(b)",
    "correct_answer": "(b)",
    "score": 1,
    "answer_options": [...]
  },

  "compliant": {
    "reasoning_response": "Looking at this thread...",
    "clean_answer_response": "(e)",
    "extracted_prediction": "(e)",
    "correct_answer": "(e)",
    "score": 1,
    "answer_options": [...]
  },

  "metadata": {
    "rule": "Close your post by replying...",
    "rule_cluster_id": 5,
    "rule_cluster_label": "civility rules",
    "subreddit_cluster_id": 2,
    "subreddit_cluster_label": "tech communities"
  },

  "few_shot": {
    "violating": [
      {
        "target_key": "train:abc123:violating",
        "subreddit": "excel",
        "thread_type": "violating",
        "correct_answer": "(b)",
        "score": 0.83
      }
    ],
    "compliant": []
  }
}
```

### Performance JSON

```json
{
  "model": "qwen3-vl-8b-instruct",
  "split": "test",
  "context": "submission-discussion",
  "phrase": "cot",
  "mode": "prefill",
  "rag": {
    "k": 4,
    "filter": "rule-cluster",
    "balance": "mixed",
    "source_split": "train",
    "retrieval_path": "output/eval/rag/test_to_train_target_comment_similarity.pt",
    "retrieval_artifact_sha256": "abc123...",
    "run_suffix": "rag-k4-rule-cluster-mixed-src-train-art-abc123def456"
  },

  "metrics": {
    "overall": {
      "total_pairs": 4166,
      "total_threads": 8332,
      "overall_accuracy": 0.815,
      "violating_accuracy": 0.85,
      "compliant_accuracy": 0.78,
      ...
    },

    "per_rule_cluster": {
      "civility rules": {
        "overall_accuracy": 0.88,
        "violating_accuracy": 0.9,
        "compliant_accuracy": 0.85,
        "count": 500
      },
      ...
    },

    "per_subreddit_cluster": {
      "tech communities": {
        "overall_accuracy": 0.82,
        "violating_accuracy": 0.85,
        "compliant_accuracy": 0.79,
        "count": 1000
      },
      ...
    }
  }
}
```

## Extending the Framework

### Adding New Models

Edit `config.py` and add to `VLLM_MODELS` or `API_MODELS`:

```python
VLLM_MODELS = {
    'my-new-model': {
        'hf_path': 'org/model-name',
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.95,
        'trust_remote_code': True,
        'max_model_len': 8192,
        'prefill_mode': 'append'
    }
}
```

### Adding New Context Types

Context strings are parsed as dash-separated flags in `config.parse_context_flags()`.
To add a new flag, edit `VALID_CONTEXT_FLAGS`, `parse_context_flags()`, and the
prompt formatting in `helpers._build_question_text()`:

```python
VALID_CONTEXT_FLAGS = {
    'none', 'submission', 'media', 'discussion', 'user', 'my_flag'
}
```

### Adding New Phrases

Edit `config.py` and add to `PHRASES`:

```python
PHRASES = {
    'my_phrase': 'Custom instruction text here.'
}
```

## Notes

- API evaluation currently uses OpenAI Flex for Stage 1 reasoning and a local
  Qwen3-VL model for Stage 2 answer extraction.
- Results and logs are grouped by `{model}/{split}/{context}/{phrase}_{mode}`;
  RAG variants are stored in child run directories under that phrase/mode path.
  baseline runs always use a `baseline/` directory because `--mode` is ignored
  for the empty baseline phrase.
