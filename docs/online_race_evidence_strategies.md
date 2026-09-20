# Online RACE Evidence and Aggregation Strategies

This explanation describes the evidence strategies implemented by
`packages/race/src/race/llm/pipelines/online_race.py`. The main distinction is
whether the model generates tokens autoregressively or evaluates an existing
sequence with teacher forcing.

## Evidence sources

Online RACE converts the attention and MLP intermediate activations at one
token position into signed Residual-Direction Alignment (RDA) evidence. Each
selected position contributes one observation to a posterior update.

The pipeline supports three evidence sources:

- output tokens produced through autoregressive generation,
- input tokens from a provided text sequence,
- replay tokens formed by appending an external generation to its prompt.

Autoregressive evidence requires generation. Input and replay evidence use
teacher forcing and do not sample new tokens from the current model.

## Autoregressive evidence

The model generates an output from the formatted prompt. Only activations for
generated tokens contribute evidence. Prompt and prefill tokens provide context
but do not update the posterior.

The number of observations depends on the generated length. If a sample reaches
the generation limit without producing an end-of-sequence token, the pipeline
records the sample in metadata but excludes it from posterior updates.

## First-token evidence

First-token evidence is a restricted autoregressive strategy. The model still
generates an output, but only the activation for the first generated token
contributes evidence.

Use this strategy to isolate the initial prediction after the prompt without
including dependencies introduced by later generated tokens. Because the
strategy depends on generation, it cannot be combined with teacher forcing.

## Full-sequence input evidence

Full-sequence input evidence uses teacher forcing. The model performs one
forward pass over a provided text sequence, and every input token contributes
an observation.

Use this strategy to analyze the evidence induced by fixed natural-language
text, source code, or another predetermined sequence. It does not produce
generated token IDs, so post-processing that depends on generated IDs is not
available.

## External-generation replay evidence

Replay evidence uses teacher forcing over a prompt combined with an assistant
output saved by an external evaluation or inference system. The current model
does not resample the answer.

By default, both prompt and appended assistant-output tokens contribute
evidence. Set `--evalscope-evidence-scope assistant` to include only the
assistant-output tokens. This option is useful when the prompt should provide
context but scoring should cover only generated output.

You can limit the external output before replay. The character limit is applied
before the token limit, and neither limit truncates the prompt.

## Token-level selection

After selecting the main evidence source, the pipeline can further restrict
generated tokens. For example, the boxed-answer mask keeps only the token range
for the final boxed answer and excludes reasoning or auxiliary text.

This mask requires generated token IDs, so it applies only to autoregressive
evidence. If no tokens remain after filtering, the sample can remain in run
statistics but does not contribute evidence.

## Posterior aggregation

Online RACE aggregates evidence by domain as the pipeline runs. For each
selected token, it reads the activation before the attention output projection
and the activation before the MLP down projection. It converts each activation
to signed RDA evidence and updates the corresponding per-layer, per-module
Normal-Inverse-Gamma posterior.

Each token is an independent observation in the sufficient statistics. A
multi-token sequence is not averaged into one sample-level observation. The
selected token range therefore determines how many posterior observations each
sample contributes.

After each generation or teacher-forcing batch, the pipeline extracts the
required activations, updates the posterior, and releases the captured
activations. Result files contain accumulated posterior statistics instead of
complete activation traces.

## Choose a strategy

| Goal | Evidence strategy |
| --- | --- |
| Analyze a complete generation trajectory | All generated tokens |
| Analyze the initial prediction after a prompt | First generated token |
| Analyze fixed text or source code | Full input sequence with teacher forcing |
| Analyze output produced by another system | External-generation replay |
| Restrict analysis to an answer span | Autoregressive evidence with a token mask |
