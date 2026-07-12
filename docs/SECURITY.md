# Security and data boundary

Digital Unconscious is local-first, not local-model. This distinction is part of the product contract.

## What stays local

- the workspace, idea backlog, research runs, prompt versions, and service state
- the local dashboard and its HTTP server
- encrypted browser credentials stored by the credential broker
- temporary provider input files after the process exits (they are deleted)

## What can leave the machine

- a text prompt or compressed behavior summary sent to the selected model
- a screenshot sent for vision interpretation
- literature and dataset queries sent to public scholarly services
- URLs opened by an explicitly started browser task
- paper text and aggregate dataset profiles supplied to Idea Lab's selected models

Using `codex exec` or `claude -p` avoids a separately managed API key; it does not make inference on-device. Their subscription services still receive the model input.

## Local CLI isolation

Ordinary model calls run in a fresh temporary directory, not the repository.

- Codex uses `--sandbox read-only`, `--ephemeral`, and a non-git temporary root.
- Claude Code uses safe mode and no tools by default.
- Image calls expose only temporary image files; Claude receives only the `Read` tool.
- Web search is enabled only for calls that request it.
- Temporary files and output schemas are deleted at the end of the call.

## Secrets

- The setup form does not accept or persist provider API keys.
- Hosted provider keys should be supplied through environment variables.
- Never commit keys to `config/pipeline.toml` or `workspace/`.
- The encrypted credential vault is for supervised browser flows; it does not export cookies or browser profiles.

## Passive observation

Before enabling screenshots:

1. Add sensitive applications to `observation.blacklist_apps`.
2. Use a dedicated desktop or user profile if the screen regularly contains secrets.
3. Prefer screenpipe or a manual log when screenshots are too broad.
4. Review the configured model route with `du models`.

No content filter can reliably remove every secret from a full screenshot. The application therefore describes the boundary honestly instead of claiming that screenshots never leave the device.

## Paper and dataset inputs

- Local paper text is sent to the selected evidence-extraction model in bounded excerpts.
- Dataset profiling is local. Model prompts contain field names and aggregate structure, but never local paths, filenames, raw rows, or example values.
- Source manifests store local paths and content hashes in the local workspace.
- PDF support is optional; image-only PDFs are rejected with an OCR instruction rather than guessed from.
- A model-provided quote becomes evidence only when deterministic code finds the exact normalized anchor in the source.

## Browser boundary

Browser task packs are supervised. They must stop at CAPTCHA, MFA, consent, terms acceptance, payments, subscription changes, or any action outside their allowlisted research task. Final publication and submission always require a human decision.

Automatic browser acquisition is not part of a daily scan or Idea Lab session. Direct paper downloads accept only bytes with a PDF signature and record a SHA-256. The app never instructs a model to bypass a paywall; lawful open access, author manuscripts, and user-authorized institutional access are the boundary.

## Reporting

Report a vulnerability privately through the repository's GitHub security reporting feature. Do not include real credentials, screenshots, or private workspace artifacts in a public issue.
