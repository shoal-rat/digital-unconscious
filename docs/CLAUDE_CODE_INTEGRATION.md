# Claude Code Integration Boundary

This repository supports Claude Code or other computer-use systems through exported task packs, not by silently taking over a personal browser profile.

## What Is Implemented

- `export-computer-task --run-id <id>` writes `computer_use_task.json`
- the task pack lists literature and dataset URLs to inspect
- login and consent moments are marked as human approval checkpoints

## What Is Not Implemented

- copying or reusing your primary Chrome profile
- storing browser cookies or passwords in the repo
- unsupervised acceptance of terms, payments, or institutional prompts

## Recommended Operating Model

- run the browser automation in a dedicated sandbox or devcontainer
- use a separate browser profile for research automation
- keep allowed domains explicit
- require human approval for logins, CAPTCHA, and consent steps
