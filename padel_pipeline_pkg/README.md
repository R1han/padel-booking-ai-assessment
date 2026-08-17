# Padel data ingestion pipeline — stage 1 (cleaning)

Identifies, records, and fixes data quality issues in the 11 source JSON files
before they are loaded into SQLite / Chroma. Nothing is fixed silently — every
issue is written to an evidence-backed ledger.

## Layout
```
run_pipeline.py              CLI orchestrator (5 stages)
pipeline/
  models.py                  pydantic schemas (structural contracts per entity)
  checks_rules.py            deterministic checks: dup IDs, FKs, enums, ranges,
                             price-rule math (round-to-5 aware), slot/booking
                             consistency, schedule overlaps
  checks_semantic.py         text-vs-field checks: court type from description,
                             coach years + languages from bio, class age ranges,
                             package numbers (compound word numbers supported)
  llm.py                     optional LLM tier (Anthropic API); auto-skipped
                             when ANTHROPIC_API_KEY is not set
  fixes.py                   resolution engine: per-issue-type policies,
                             confidence thresholds, dependency ordering
                             (court type before price inference), post-fix
                             invariant verification
  ledger.py                  Issue dataclass + Ledger (JSON/CSV export, summary)
```

## Run
```
pip install pydantic requests
python run_pipeline.py --input <dir with the 11 json files> --output <out dir>
# optional LLM tier:
export ANTHROPIC_API_KEY=sk-ant-...   # then run as above; --no-llm to disable
```

## Outputs
- `<out>/cleaned/*.json`     corrected datasets, same shape as input
- `<out>/issue_ledger.json`  every issue: detected/corrected value, evidence,
- `<out>/issue_ledger.csv`   confidence, action (auto_fixed / quarantined /
                             validated_ok)
- `<out>/report.md`          run summary incl. quarantined items

## Baked-in dataset facts / owner decisions
- Weekend = Fri/Sat (verified against slot prices: 1664/1680 vs 1224 for Sat/Sun)
- slot_price = round5(court_price_per_hour * rule_multiplier) — used to recover
  missing/sentinel court prices by inversion
- Description wins over structured `type` for courts
- Null max_age on adult classes = no upper limit (valid)
- 90-min bookings on one 60-min slot = intentional slot overhang (not flagged)
- Exit code is non-zero if post-fix verification fails
