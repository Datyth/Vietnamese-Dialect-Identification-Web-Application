# Repository Implementation Guide

Implement each request with the simplest correct solution. This is a lightweight
Vietnamese dialect identification project; avoid architecture or infrastructure
that the current task does not require.

Before implementing any task, read PLAN.md and keep changes aligned with the current phase.

## Before Coding

- Inspect the repository layout, naming, dependencies, scripts, and tests.
- Follow existing conventions. If none exist, use the smallest reasonable
  structure and create only directories needed by the task.
- Inspect real dataset examples before assuming schemas, labels, or speaker IDs.

## Implementation Rules

- Prefer plain Python functions, short modules, readable control flow, and clear
  names.
- Keep changes limited to the requested feature. Do not refactor, rename, or move
  unrelated code.
- Avoid factories, registries, plugin systems, dependency injection, config
  frameworks, and deep class hierarchies unless already required by the codebase.
- Establish a working baseline before optimizing. A simple script is preferable
  to a complex CLI.
- Keep important parameters in one clear config or constants location. Set seeds
  for data splitting and training, make output paths explicit, and do not silently
  overwrite important outputs.
- Validate required inputs and fields. Raise clear, specific errors for missing
  files, fields, or dependencies. Do not hide failures with broad exception
  handling.
- Do not guess when information is unavailable. Record assumptions, missing
  speaker IDs, unclear labels, and locally unverified behavior in the report.

## Project Scope

Support only three classes: `Northern`, `Central`, and `Southern`.

The intended pipeline is metadata preparation, audio preprocessing, an MFCC
baseline, a lightweight CNN, evaluation, and a simple web app. A
PhoWhisper-base experiment is optional and should be added only when requested.

Do not add province-level classification, speaker or hometown prediction,
distributed training, databases, background workers, cloud or production
deployment, Docker, ONNX export, complex MLOps or monitoring, large
hyperparameter searches, or multiple pretrained models unless explicitly
requested.

## Dependencies

Before adding a dependency:

1. Check whether an existing dependency or the Python standard library is enough.
2. Add only the smallest suitable dependency.
3. Document why it is needed, where it is used, and why a simpler option is
   insufficient.

Do not add a heavy dependency for a small task.

## Organization

Use existing repository conventions. If no convention exists, choose only the
needed parts of a simple layout such as `src/`, `data/`, `features/`, `models/`,
`training/`, `evaluation/`, `inference/`, `utils/`, `outputs/`, and `reports/`.
Do not create empty scaffolding.

## Verification And Reporting

- Include a minimal relevant check: a unit test, dry run, 3-5 sample run, shape
  check, metrics check, or generated-report check.
- Never claim success for a check that was not run. State why it was not run.
- After each implementation, create or update
  `reports/implementation_report.md` using its required sections.
- Keep the report short and concrete. Include changed files, scope, decisions,
  exact run commands, outputs, verification results, limitations, and reviewer
  priorities.

The final response must list the summary, files changed, commands run, checks
performed, report path, and any uncertainty or limitation.
