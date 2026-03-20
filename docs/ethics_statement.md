# Ethics Statement

**Project:** A Multi-Component NLP System for Fake News and Misinformation Detection
**Team:** Rian Cortez, Ryence Cortez, Ranz Cuarto, Mark Darius David
**Course:** 6INTELSY — Intelligent Systems, Holy Angel University
**Date:** March 2026

---

## Why this matters

Building a classifier that labels political speech as true or false is not a neutral technical exercise. Errors in either direction carry real consequences. A false positive flags legitimate speech as misinformation. A false negative lets false claims pass undetected. Both outcomes affect real people, and the fact that a model produces them automatically makes it easy to overlook how often they happen. This document is our attempt to be honest about those risks and what we are doing about them.

---

## Risk 1: False positives flagging legitimate content

This is our highest-priority concern. The LIAR dataset is a difficult 6-class problem and our current models achieve Macro-F1 scores around 0.22 to 0.24, which means a large portion of predictions are wrong. In a real deployment, those wrong predictions would mean real statements being incorrectly labeled as false.

What we are doing about it: the system is designed as a tool for human reviewers, not an autonomous decision-maker. Every prediction comes with a confidence score. The bandit agent learned to set a conservative threshold of 0.8, meaning the system will abstain rather than guess when it is not confident. We report calibration curves alongside F1 scores so anyone using the system can see how well the confidence scores actually reflect accuracy.

---

## Risk 2: Political bias in the training data

The LIAR dataset has more Republican-affiliated statements than Democrat-affiliated ones in the training set (4,497 vs 3,336). This is not something we introduced — it reflects how PolitiFact sourced its data — but it is something we are responsible for tracking. A model trained on this imbalance may learn to associate certain political signals with certain veracity labels in ways that are not actually about the truth of the statement.

What we are doing about it: we ran per-slice F1 analysis broken down by party affiliation and entity type. The NER analysis already shows that statements mentioning specific locations and organizations perform worse, which is worth investigating further. In Week 3 we will run a full party-level slice analysis on DistilBERT predictions and report whatever we find, including results that reflect poorly on the system.

---

## Risk 3: Misuse as a censorship tool

A misinformation classifier that works reasonably well is a dual-use tool. The same model could be used by a bad actor to automatically suppress political content they disagree with. We cannot fully prevent this, but we can make our intentions clear and limit how the system is packaged.

What we are doing about it: the repository includes an explicit intended-use statement limiting the system to research and fact-checker assistance. The model license prohibits commercial and governmental censorship use. No public API or deployment endpoint will be released. The demo is a local script only.

---

## Data and privacy

Every statement in the LIAR dataset is a public utterance made by a public figure in an official capacity, sourced from PolitiFact's public archive. No private communications, personal data, or user-generated content is included. We do not collect any data from users of this system.

---

## What we checked in Week 2

- Verified class distribution is consistent across train, validation, and test splits with no leakage
- Documented the Republican vs Democrat speaker imbalance in the training set
- Ran NER-based slice analysis on the test set — location and organization mentions correlate with lower performance
- Confirmed the dataset contains no PII beyond publicly available speaker names

---

## What we still need to check

- Full party-level slice analysis on DistilBERT predictions (Week 3)
- Calibration curve analysis once DistilBERT fine-tuning is complete
- Error analysis on the worst-performing label classes (pants-fire in particular)

---

## Intended use

This system was built as a course project for 6INTELSY at Holy Angel University. It is not production-ready and should not be treated as one. If you are using this system, use it to assist your own judgment — not to replace it.
