# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
python src/train.py
VOTES_WORKER_URL=https://mtcp-votes.mr-tony-82.workers.dev python src/predict.py
```

## Conventions & Patterns

### Feature/model changes require a committed retrain

Any change to `src/features.py` or `src/train.py` is **not complete** until:
1. `python src/train.py` runs successfully
2. `python src/predict.py` runs successfully
3. All generated files are committed alongside the code change:
   - `model/model.joblib`
   - `model/model_intraday.joblib`
   - `data/feature_snapshots.json`
   - `data/weather_cache.json`
   - `docs/predictions.json`

The model files must be trained on the same feature set `predict.py` uses — shipping code without updated models breaks the GitHub Actions predict workflow.

### Per-model data consistency

Each of the three ML models (IFS, NBM, Ensemble) must use its **own** data source end-to-end — both for historical features and forecast features. Do not share or average weather/precipitation data across models unless explicitly instructed.

| Feature | IFS model | NBM model | Ensemble model |
|---|---|---|---|
| Historical weather cache | `hist` (best_match) | `hist_n` (ncep_nbm_conus) | `hist_e` (ensemble cache) |
| Forecast precipitation | IFS forecast | NBM forecast | Average of 6 NWP models |
| `precip_2d`, `rain_3d_to_7d_mm`, `hours_since_rain` | From IFS cache | From NBM cache | From ensemble cache |

**Intentional exceptions — these ARE shared across all three models:**

- **Soil features** (`soil_moisture`, `soil_moisture_deep`, `soil_temp_*`): sourced from `best_match`/`ecmwf_ifs025` only, because other NWP models don't reliably provide soil data. See `_average_forecasts()` comment in `predict.py`.
- **`prior_report_score`**: derived from MTBProject trail condition report, fetched once per trail and shared across all three models.
- **Training labels / user vote scores**: from Cloudflare KV (user feedback), shared across all models — these are ground truth, not model-specific inputs.

If you find yourself writing `precip_2d_canon` or averaging weather cache values across models, stop and re-read this rule.
