# Implementation Status Against `autonomous_research_pipeline_v2.pdf`

## Implemented In This Repository

- Daily idea capture from exported text or `jsonl` logs
- Literature scouting from open providers
- Open PDF download attempts when direct links are available
- Structured paper metadata with heuristic claims, methods, findings, and dataset hints
- Feasibility scoring with go/review/archive output
- Open dataset discovery
- Local CSV descriptive analysis
- Processed-data copy plus provenance metadata
- Reproducibility rerun check
- Manuscript Markdown generation
- BibTeX generation
- Simple generated manuscript PDF
- Review scoring with revision history
- Learning signal aggregation
- Supervised Claude Code computer-use task export

## Not Fully Implemented Here

- Passive 24/7 screen observation
- Institutional credential vault and encrypted reuse
- Unsupervised browser login flows against personal Chrome profiles
- Paywalled portal navigation without human approval
- LLM-backed writer/reviewer agents using Claude models directly
- Statistical testing with p-values, effect sizes, and confidence intervals for arbitrary studies
- Journal-specific camera-ready typesetting
- Final submission workflow and approval queue
- Prompt evolution and domain-RAG expansion loops

## Why Those Items Are Still Separate

The remaining items are not just more Python code inside this repo. They require one or more of:

- external model providers and prompt orchestration
- a sandboxed browser automation environment
- sensitive credential handling
- institution-specific portal behavior
- legal and security controls around account use

That boundary is deliberate. The repository now covers the parts that are realistically safe and buildable locally without pretending the external automation pieces are solved when they are not.
