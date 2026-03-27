# Ethics Statement

**Project:** A Multi-Component NLP System for Fake News and Misinformation Detection
**Team:** Rian Cortez, Ryence Cortez, Ranz Cuarto, Mark Darius David
**Course:** 6INTELSY — Intelligent Systems, Holy Angel University
**Date:** March 2026

---

## Why this matters

Building a classifier that labels political speech as true or false is not a neutral technical exercise. Errors in either direction carry real consequences. A false positive flags legitimate speech as misinformation. A false negative lets false claims pass undetected. Both outcomes affect real people, and the fact that a model produces them automatically makes it easy to overlook how often they happen. This document is our attempt to be honest about those risks and what we did about them.

---

## Risk 1: False positives flagging legitimate content

This is our highest-priority concern. The LIAR dataset is a difficult 6-class problem and our final DistilBERT model achieves a test Macro-F1 of 0.2602, which means a significant portion of predictions are still wrong. In a real deployment, those wrong predictions would mean real statements being incorrectly labeled as misinformation.

What we did about it: the system is designed as a tool for human reviewers, not an autonomous decision-maker. Every prediction comes with a confidence score and a full probability distribution across all six classes. The bandit agent learned to set a conservative threshold of 0.8, meaning the system will abstain rather than guess when it is not confident. Calibration curves confirm the model is overconfident above 0.5 probability, which reinforces why the conservative threshold is necessary.

---

## Risk 2: Political bias in the training data

The LIAR dataset has more Republican-affiliated statements than Democrat-affiliated ones in the training set (4,497 vs 3,336). This is not something we introduced — it reflects how PolitiFact sourced its data — but it is something we are responsible for tracking. A model trained on this imbalance may learn to associate certain political signals with certain veracity labels in ways that are not actually about the truth of the statement.

What we did about it: we ran a full party-level slice analysis on DistilBERT predictions. Republican-affiliated statements achieve a Macro-F1 of 0.2231, substantially below the overall test Macro-F1 of 0.2602, while Democrat-affiliated statements achieve 0.2715, above the overall average. This disparity is documented in the Model Card without softening. NER slice analysis further shows that statements mentioning specific locations (GPE Macro-F1: 0.2011) and organizations (ORG Macro-F1: 0.2068) are consistently harder to classify, suggesting performance varies by the contextual framing of political content as well as party affiliation.

---

## Risk 3: Misuse as a censorship tool

A misinformation classifier that works reasonably well is a dual-use tool. The same model could be used by a bad actor to automatically suppress political content they disagree with. We cannot fully prevent this, but we can make our intentions clear and limit how the system is packaged.

What we did about it: the repository includes an explicit intended-use statement limiting the system to research and fact-checker assistance. The model license prohibits commercial and governmental censorship use. No public API or deployment endpoint has been released. The demo is a local script only. The party-level performance disparity identified in Risk 2 further underscores why autonomous deployment without human oversight would be harmful regardless of intent.

---

## Data and privacy

Every statement in the LIAR dataset is a public utterance made by a public figure in an official capacity, sourced from PolitiFact's public archive. No private communications, personal data, or user-generated content is included. We do not collect any data from users of this system.

---

## What we confirmed in Week 2

- Verified class distribution is consistent across train, validation, and test splits with no leakage
- Documented the Republican vs Democrat speaker imbalance in the training set
- Ran NER-based slice analysis on the test set — location and organization mentions correlate with lower performance
- Confirmed the dataset contains no PII beyond publicly available speaker names

---

## What we confirmed in Week 3

- Full party-level slice analysis on DistilBERT predictions confirmed Republican F1 gap of ~0.05 below overall
- Calibration curve analysis confirmed overconfidence above 0.5 probability — conservative threshold of 0.8 is justified
- Error analysis on confusion matrix confirmed pants-fire is the hardest class (F1: 0.20) and is most frequently confused with false
- Bandit agent re-evaluated with real DistilBERT outputs — converged on threshold 0.8, outperforms fixed baseline

---

## Intended use

This system was built as a course project for 6INTELSY at Holy Angel University. It is not production-ready and should not be treated as one. If you are using this system, use it to assist your own judgment — not to replace it.
