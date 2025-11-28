# Implementation Notes

## Overview

This evaluation framework is designed to evaluate Vision Language Models (VLMs) on Reddit moderation tasks. It follows the same architecture as `/data3/zkachwal/zeroshot-pgt` but is customized for multimodal Reddit moderation evaluation.

## Architecture

### Three Main Components

1. **config.py**: All configurations (models, contexts, phrases, paths)
2. **helpers.py**: All utility functions (data loading, prompting, evaluation, metrics)
3. **evaluate.py**: Orchestration logic (minimal, DRY)

This separation ensures:
- Easy extension (add models/contexts/phrases in config)
- Maintainability (helpers are organized by function)
- Clarity (evaluate.py is just pipeline orchestration)

## Key Design Decisions

### 1. Multimodal Input Handling

Following zeroshot-pgt pattern:
- **Image paths** stored in dataset JSON
- **Lazy loading**: PIL Images loaded just before inference (not during dataset loading)
- **vLLM format**:
  ```python
  {
    "prompt": "text_prompt_string",
    "multi_modal_data": {"image": [PIL_Image, ...]},
    "multi_modal_uuids": {"image": ["uuid_0", "uuid_1", ...]}
  }
  ```

### 2. Two-Stage Evaluation

**Stage 1: Reasoning Generation**
- Temperature: 0.0 (for reproducibility)
- Max tokens: 512
- Output: Full reasoning trace

**Stage 2: Clean Answer Extraction**
- Input: Stage 1 prompt + response + "Final Choice:"
- Temperature: 0.0
- Max tokens: 10
- Stop string: "\n" (with include_stop_str_in_output=True)
- Output: Single choice like "(a)", "(b)", etc.

### 3. Dual Thread Evaluation

Each thread pair produces **two predictions**:
- **Moderated thread**: Real moderator comment + thread context
- **Unmoderated thread**: Counterfactual thread without intervention

Both evaluated simultaneously for efficiency.

### 4. Context System (Placeholder)

Context types control prompt construction:
- `minimal`: Submission + thread only
- `rule_only`: Just matched rule
- `thread_with_rule`: Thread + submission + matched rule
- `thread_with_all_rules`: Thread + submission + all rules
- `full`: Everything including media and metadata

**Current status**: Structure in place, actual formatting in `helpers._build_question_text()` is PLACEHOLDER.

### 5. Metrics Tracking

Three levels of accuracy:
1. **Overall**: All threads (moderated + unmoderated)
2. **Moderated-only**: Just moderated threads
3. **Unmoderated-only**: Just unmoderated threads

Plus breakdowns by:
- **Rule cluster**: Performance on each rule type cluster
- **Subreddit cluster**: Performance on each subreddit type cluster

## File Structure

```
reddit-mod-collection-pipeline/
├── eval/
│   ├── config.py                 # Configurations
│   ├── helpers.py                # Utility functions
│   ├── evaluate.py               # Main script
│   ├── __init__.py               # Package init
│   ├── README.md                 # User documentation
│   └── IMPLEMENTATION_NOTES.md   # This file
├── output/
│   └── eval/                     # Results (auto-created)
│       └── {model}/{split}/{context}/{phrase}_{mode}/
│           ├── reasoning_TIMESTAMP.json
│           └── performance_TIMESTAMP.json
└── logs/
    └── eval/                     # Logs (auto-created)
        └── {model}/{split}/{context}/{phrase}_{mode}/
            └── evaluation_TIMESTAMP.log
```

## TODO: Next Steps

### High Priority

1. **Implement Context Formatting**
   - Flesh out `helpers._build_question_text()`
   - Format submission content (title, selftext, media references)
   - Format thread comments (hierarchical structure, usernames, timestamps)
   - Format rules (matched rule only vs. all rules)
   - Format multiple choice options consistently

2. **Test with Real Data**
   - Run debug mode on actual test split
   - Verify image loading works
   - Verify prompt formatting looks correct
   - Test answer extraction with actual model outputs

3. **Add API Model Support**
   - Implement `_evaluate_two_stage_api()` in helpers
   - Add Claude Batch API integration
   - Add OpenAI Batch API integration
   - Handle image base64 encoding for APIs

### Medium Priority

4. **Add n>1 Sampling**
   - Support multiple responses per thread (like zeroshot-pgt)
   - Add majority voting aggregation
   - Track individual response metrics vs. aggregated

5. **Add Visualization Tools**
   - Script to plot accuracy by cluster
   - Confusion matrices for rule types
   - Error analysis helpers

6. **Batch Processing**
   - Support for processing large datasets in batches
   - Checkpoint/resume functionality
   - Progress tracking

### Low Priority

7. **Advanced Features**
   - Support for few-shot examples
   - Support for rule-specific prompts
   - Support for cluster-specific evaluation
   - Integration with analysis scripts

## Design Patterns from zeroshot-pgt

### What We Adopted

✅ **Three-file architecture** (config, helpers, evaluate)
✅ **Two-stage evaluation** (reasoning + clean answer)
✅ **Lazy image loading** (paths in data, load before inference)
✅ **vLLM input format** (prompt + multi_modal_data + uuids)
✅ **Chat template with AutoProcessor**
✅ **Phrase modes** (prefill, prompt)
✅ **Directory organization** (output/{model}/{split}/...)
✅ **Logging setup** (file + console handlers)
✅ **Result saving** (reasoning JSON + performance JSON)

### What We Adapted

🔄 **Dataset loading**: Custom for Reddit JSON structure
🔄 **Prompt building**: Multiple choice format for moderation
🔄 **Dual evaluation**: Both moderated and unmoderated threads
🔄 **Metrics**: Accuracy-based, not F1 (multiclass problem)
🔄 **Context system**: Configurable information exposure
🔄 **Cluster tracking**: Rule clusters + subreddit clusters

### What We Simplified

⚡ **No n>1 sampling yet**: Start with single response per thread
⚡ **No batch API yet**: vLLM only to start
⚡ **No confidence scores yet**: Binary correct/incorrect

## Testing Strategy

1. **Unit test imports**: ✅ Done
2. **Test help command**: ✅ Done
3. **Debug mode test**: Next step - run with --debug flag
4. **Validate outputs**: Check reasoning.json and performance.json structure
5. **Manual review**: Look at actual prompts and model responses
6. **Full evaluation**: Run on small subset without --debug

## Example Command

```bash
# Debug run (5 thread pairs)
python3 eval/evaluate.py \
    --model qwen25-vl-7b \
    --split test \
    --context minimal \
    --phrase baseline \
    --mode prefill \
    --debug

# Real run
python3 eval/evaluate.py \
    --model qwen25-vl-7b \
    --split test \
    --context thread_with_rule \
    --phrase cot \
    --mode prefill
```

## Notes for Future Developers

### Adding New Models

1. Add config to `config.py` (VLLM_MODELS or API_MODELS)
2. If special formatting needed, update `helpers._build_multimodal_content()`
3. Test with --debug first

### Adding New Context Types

1. Add config to `config.py` (CONTEXT_TYPES)
2. Implement formatting in `helpers._build_question_text()`
3. Document what information is included/excluded

### Adding New Phrases

1. Add to `config.py` (PHRASES)
2. No code changes needed if using standard modes
3. For custom behavior, update `helpers._build_single_prompt()`

### Debugging Tips

- Use `--debug` flag to test with 5 thread pairs
- Check logs in `eval/logs/{model}/{split}/{context}/{phrase}/`
- Inspect `reasoning_*.json` to see actual model outputs
- Print statements in helpers are logged automatically

## Dependencies

Required packages:
- `vllm` (for vLLM models)
- `transformers` (for AutoProcessor)
- `pillow` (for image loading)
- `zstandard` (for compressed dataset loading)

Standard library:
- `json`, `logging`, `pathlib`, `argparse`, `collections`, `datetime`

## Known Limitations

1. **Context formatting is placeholder** - needs implementation
2. **API models not implemented** - vLLM only for now
3. **No n>1 sampling** - single response per thread only
4. **No few-shot support** - zero-shot only
5. **Fixed stop string** - assumes single-letter answers

These are intentional to get MVP working first.
