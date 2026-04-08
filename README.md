---
title: Support Ticket Openenv
emoji: 👀
colorFrom: yellow
colorTo: pink
sdk: docker
pinned: false
---

# Support Ticket Triage OpenEnv

A real-world OpenEnv environment where an AI agent performs email and support-ticket triage.

---

## Environment Description

This environment simulates a customer support workflow.

The agent receives a support ticket and must:
- classify the issue
- set priority
- detect spam
- choose the correct final support action

---

## Difficulty Levels

- **Easy** → classification only  
- **Medium** → classification + priority  
- **Hard** → classification + priority + spam + action  

---

## Action Space

The agent can perform:
- `classify_ticket`
- `set_priority`
- `mark_spam`
- `choose_action`
- `submit`

---

## Observation Space

Each observation includes:
- task ID  
- difficulty  
- subject  
- message  
- action history  
- allowed actions  
- steps taken  
- steps remaining  
- current reward  
- done flag  
- feedback  

---

## State (Hidden)

Internal state contains:
- correct answers (ground truth)  
- agent predictions  
- reward  
- steps taken  
- done status  

---

## Reward Function

Scores range from **0.0 to 1.0**

### Easy
- classification → 1.0  

### Medium
- classification → 0.6  
- priority → 0.4  

### Hard
- category → 0.3  
- priority → 0.2  
- spam → 0.2  
- action → 0.3  

---

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
=======

