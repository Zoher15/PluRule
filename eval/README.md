# Reddit Moderation Evaluation Framework

Evaluate Vision Language Models (VLMs) on Reddit moderation tasks using clustered datasets with multimodal inputs.

## Features

- **Multiple Model Support**: vLLM models (Qwen, LLaVA, Llama-Vision) and API models (Claude, GPT-4V)
- **Configurable Contexts**: Control what information is exposed in prompts (rules, threads, media, metadata)
- **Phrase Variations**: Test different prompting strategies (baseline, chain-of-thought, etc.)
- **Two-Stage Evaluation**: Generate reasoning first, then extract clean answers
- **Comprehensive Metrics**: Overall, per-rule-cluster, and per-subreddit-cluster accuracy

## Directory Structure

```
reddit-mod-collection-pipeline/
├── eval/
│   ├── config.py           # Model, context, and phrase configurations
│   ├── helpers.py          # Utility functions for data loading, prompting, evaluation
│   └── evaluate.py         # Main evaluation script
├── output/
│   └── eval/               # Evaluation results (reasoning + performance JSONs)
│       └── {model}/
│           └── {split}/
│               └── {context}/
│                   └── {phrase}_{mode}/
│                       ├── reasoning_TIMESTAMP.json
│                       └── performance_TIMESTAMP.json
└── logs/
    └── eval/               # Evaluation logs
        └── {model}/
            └── {split}/
                └── {context}/
                    └── {phrase}_{mode}/
                        └── evaluation_TIMESTAMP.log
```

## Installation

Requires dependencies from the main pipeline plus vLLM:

```bash
pip install vllm transformers pillow
```

## Usage

### Basic Usage

```bash
# Evaluate Qwen2.5-VL-7B on test set with thread+rule context
python eval/evaluate.py \
    --model qwen25-vl-7b \
    --split test \
    --context thread_with_rule \
    --phrase cot \
    --mode prefill

# Debug mode (only 5 thread pairs)
python eval/evaluate.py \
    --model qwen25-vl-7b \
    --split test \
    --context minimal \
    --phrase baseline \
    --mode prefill \
    --debug
```

### Arguments

- `--model, -m`: Model to evaluate
  - vLLM: `qwen25-vl-7b`, `qwen25-vl-72b`, `llava-onevision-7b`, `llama32-vision-11b`
  - API: `claude-sonnet-4`, `gpt-4o` (not yet implemented)

- `--split, -s`: Dataset split (`train`, `val`, `test`)

- `--context, -c`: Context type (what to include in prompts)
  - `minimal`: Submission + thread only (no rules)
  - `rule_only`: Just the matched rule text
  - `thread_with_rule`: Thread + submission + matched rule
  - `thread_with_all_rules`: Thread + submission + all subreddit rules
  - `full`: Everything (thread + submission + all rules + metadata + media)

- `--phrase, -p`: Prompting phrase
  - `baseline`: No additional phrase
  - `cot`: "Let me think step by step."
  - `analyze`: "Let me carefully analyze this content."
  - `artifacts`: "I will look for violations and rule-breaking behavior."
  - `rules`: "I will compare this against the subreddit rules."

- `--mode`: Phrase injection mode
  - `prefill`: Append phrase after chat template (default)
  - `prompt`: Append phrase to the question text

- `--cuda`: CUDA device IDs (default: `"0"`)
- `--debug`: Run with only 5 thread pairs for testing
- `--override`: Overwrite existing results

## Output Format

### Reasoning JSON

Each thread pair produces two predictions (moderated and unmoderated):

```json
{
  "mod_comment_id": "fdzc60l",
  "subreddit": "excel",
  "submission_id": "en8cqn",

  "moderated": {
    "reasoning_response": "Let me analyze this comment...",
    "clean_answer_response": "(b)",
    "extracted_prediction": "(b)",
    "correct_answer": "(b)",
    "score": 1,
    "answer_options": [...]
  },

  "unmoderated": {
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
  }
}
```

### Performance JSON

```json
{
  "model": "qwen25-vl-7b",
  "split": "test",
  "context": "thread_with_rule",
  "phrase": "cot",
  "mode": "prefill",

  "metrics": {
    "overall": {
      "total_pairs": 4166,
      "total_threads": 8332,
      "overall_accuracy": 0.815,
      "moderated_accuracy": 0.85,
      "unmoderated_accuracy": 0.78,
      ...
    },

    "per_rule_cluster": {
      "civility rules": {
        "overall_accuracy": 0.88,
        "moderated_accuracy": 0.9,
        "unmoderated_accuracy": 0.85,
        "count": 500
      },
      ...
    },

    "per_subreddit_cluster": {
      "tech communities": {
        "overall_accuracy": 0.82,
        "moderated_accuracy": 0.85,
        "unmoderated_accuracy": 0.79,
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

Edit `config.py` and add to `CONTEXT_TYPES`:

```python
CONTEXT_TYPES = {
    'my_context': {
        'description': 'Custom context description',
        'include_submission': True,
        'include_thread': True,
        'include_matched_rule': True,
        'include_all_rules': False,
        'include_media': False,
        'include_metadata': False
    }
}
```

Then implement context formatting in `helpers._build_question_text()`.

### Adding New Phrases

Edit `config.py` and add to `PHRASES`:

```python
PHRASES = {
    'my_phrase': 'Custom instruction text here.'
}
```

## TODO

- [ ] Implement actual context formatting in `helpers._build_question_text()`
- [ ] Add API model support (Claude Batch API, GPT-4V)
- [ ] Add support for n>1 sampling with majority voting
- [ ] Add visualization scripts for results analysis
- [ ] Add batch processing for large-scale evaluation
