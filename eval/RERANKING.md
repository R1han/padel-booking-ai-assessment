# Reranking: measured, and it does not help here

Challenge 6 asks for a reranking stage demonstrated to improve results on the eval set.
The stage is built (`rerank()` in `app/services/retrieval.py`), it is measured, and the
honest result is that **it makes retrieval worse on this corpus**. It therefore ships
disabled, with the evidence below and a single env var to turn it back on.

## Setup

Listwise rerank: the RRF-fused top 20 candidates are sent to `LLM_RERANKER` with the
question, and the model returns them reordered; the top 6 are kept. It can only reorder,
never shrink, the candidate set — an earlier version was told to drop irrelevant
candidates and collapsed six results to one.

Measured over all 38 eval queries. Recall alone saturates at a generous `k`, so
precision@1 and MRR were added to detect whether reranking puts the *best* record first.

## Results

| Configuration | Recall | P@1 | MRR | Refusal acc. | Mean cost | p95 latency |
| --- | --- | --- | --- | --- | --- | --- |
| **Hybrid + RRF, no rerank** | **1.000** | **0.789** | **0.835** | **0.974** | **$0.00205** | **7598 ms** |
| \+ LLM rerank, 240-char parent snippet | 1.000 | 0.737 | 0.800 | 0.947 | $0.00224 | 8393 ms |
| \+ LLM rerank, 480-char parent snippet | 0.987 | 0.737 | 0.800 | 0.974 | $0.00240 | 9171 ms |
| \+ LLM rerank, **matched chunk** | 0.975 | **0.650** | **0.717** | 0.979 | $0.00227 | 6439 ms |

Four configurations, all pointing the same way. Reranking loses, and the variant built
specifically to fix its suspected weakness loses hardest.

Reranking cost 5 points of precision@1 and 3.5 points of MRR, added roughly $0.0002 per
query, and pushed p95 latency from inside the 8s target to outside it. Giving the
reranker twice the context per candidate did not recover the loss and cost a further
800 ms.

## Why it loses

1. **The baseline is already strong.** Reciprocal Rank Fusion over a semantic ranking and
   a lexical ranking is a genuine ensemble: the two retrievers fail in uncorrelated ways,
   and RRF needs no score calibration between them. A single model re-reading truncated
   snippets has strictly less information than the two retrievers had.
2. **The candidate set is small.** Fusing to 20 and keeping 6 leaves little room to
   improve; the relevant record is usually already inside the top few.
3. **Snippets are the wrong unit for policies.** The hardest queries are policy questions,
   where the answer sits in one numbered clause of a ~10 KB document. A leading snippet
   frequently does not contain the deciding sentence, so the reranker is judging on
   evidence that does not include the answer.
4. **The corpus punishes lexical judgement.** "rain" or "weather" appears in 37 of the 40
   policies. Asked to rank them, the model has little to separate them by and its ordering
   drifts.

## What would be worth trying next

A cross-encoder (for example `bge-reranker-base`) scores the full query-document pair
rather than re-reading snippets, which addresses cause 3 directly. It was not adopted here
because it pulls in `torch` and a few hundred MB of model weights for a stage that the
measurement says is not currently the bottleneck — the clean-clone install requirement
weighed against it. Chunk-level reranking, where policy sections are ranked individually
rather than by their parent document, is the cheaper version of the same idea.

## Reproducing

```bash
RERANK_ENABLED=false python -m app.eval --input eval/queries.json --output eval/results/rerank_off.json
RERANK_ENABLED=true  python -m app.eval --input eval/queries.json --output eval/results/rerank_on.json

python -m app.score --results eval/results/rerank_off.json --gold eval/gold.json
python -m app.score --results eval/results/rerank_on.json  --gold eval/gold.json
```
