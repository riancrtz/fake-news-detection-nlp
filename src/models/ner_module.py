"""
ner_module.py
Named Entity Recognition pipeline using spaCy.
Extracts PERSON, ORG, and GPE entities from statements
for provenance analysis and slice-level error analysis.
"""

import spacy
from collections import Counter
from sklearn.metrics import f1_score

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """
    Extract named entities from a single statement.
    Returns a dict with PERSON, ORG, GPE, and OTHER entity lists.
    """
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": [], "OTHER": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities["OTHER"].append(ent.text)
    return entities


def run_ner_pipeline(df):
    """
    Run NER over a full dataframe of statements.
    Adds entity columns and boolean presence flags.
    """
    df = df.copy()
    df['entities']   = df['statement'].apply(extract_entities)
    df['has_person'] = df['entities'].apply(lambda x: len(x['PERSON']) > 0)
    df['has_org']    = df['entities'].apply(lambda x: len(x['ORG']) > 0)
    df['has_gpe']    = df['entities'].apply(lambda x: len(x['GPE']) > 0)
    return df


def slice_analysis(df, pred_col='pred', label_col='label_id'):
    """
    Run slice analysis by entity presence.
    Reports Macro-F1 for statements with and without each entity type.
    """
    results = {}
    for col, label in [('has_person', 'PERSON'),
                        ('has_org', 'ORG'),
                        ('has_gpe', 'GPE')]:
        with_ent    = df[df[col]]
        without_ent = df[~df[col]]
        results[label] = {
            "with_entity": {
                "macro_f1": round(f1_score(
                    with_ent[label_col], with_ent[pred_col],
                    average='macro'), 4),
                "n": len(with_ent)
            },
            "without_entity": {
                "macro_f1": round(f1_score(
                    without_ent[label_col], without_ent[pred_col],
                    average='macro'), 4),
                "n": len(without_ent)
            }
        }
    return results
