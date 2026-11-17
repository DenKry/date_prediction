"""Build the pair-level modelling table from the raw MixMoSa CSV files.

One row = one 8-minute speed date (man, woman, event). Only information that is
known *before* the date is used as input (profile questionnaire, photo ratings,
event metadata), so the task is "predict compatibility in advance".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "mixmosa" / "csv"
PROCESSED = ROOT / "data_processed"

USER_NUMERIC = [
    "age", "iq_share_correct", "agreeableness", "conscientiousness", "extraversion",
    "neuroticism", "openness", "avg_physical_attractiveness_rating", "height_inches",
    "income_bucket", "nicotine_score", "alcohol_score", "marijuana_score",
    "has_used_psychedelics", "behavioral_health_score", "sex_partners_score",
    "get_along_well_with_family_score", "political_tolerance_score",
    "ethnicity_importance_score", "religion_importance_score", "politics_importance_score",
    "num_children_wanted_score", "already_has_kids", "casual_sex_score",
    "hard_work_and_success_belief_score",
]
SHORT = {
    "iq_share_correct": "iq", "avg_physical_attractiveness_rating": "attract",
    "height_inches": "height", "income_bucket": "income",
    "behavioral_health_score": "health", "get_along_well_with_family_score": "family",
    "political_tolerance_score": "pol_tolerance", "ethnicity_importance_score": "eth_importance",
    "religion_importance_score": "rel_importance", "politics_importance_score": "pol_importance",
    "num_children_wanted_score": "kids_wanted", "hard_work_and_success_belief_score": "hard_work",
    "sex_partners_score": "sex_partners", "casual_sex_score": "casual_sex",
    "nicotine_score": "nicotine", "alcohol_score": "alcohol", "marijuana_score": "marijuana",
    "has_used_psychedelics": "psychedelics", "already_has_kids": "has_kids",
}
# traits for which |man - woman| (similarity) is a sensible compatibility feature
SIMILARITY = ["age", "iq", "agreeableness", "conscientiousness", "extraversion", "neuroticism",
              "openness", "attract", "income", "alcohol", "marijuana", "health", "kids_wanted",
              "casual_sex", "pol_tolerance", "hard_work", "sex_partners"]


def load_users() -> pd.DataFrame:
    uf = pd.read_csv(RAW / "user_features.csv")
    u = uf[["user_id", "attracted_to"] + USER_NUMERIC].rename(columns=SHORT).set_index("user_id")
    u["bisexual"] = (u.pop("attracted_to") == "everyone").astype(float)
    # extra photo-rating statistics (rater agreement / number of rated photos)
    pr = pd.read_csv(RAW / "picture_ratings.csv")
    pr = pr[pr.fail_reason.isna()]
    stats = pr.groupby("user_id").rating.agg(attract_std="std", n_ratings="count")
    u = u.join(stats)
    u["questionnaire_missing"] = u["iq"].isna().astype(float)
    return u


def build_pairs() -> pd.DataFrame:
    it = pd.read_csv(RAW / "interactions.csv")
    ev = pd.read_csv(RAW / "events.csv").set_index("id")
    users = load_users()
    ucols = list(users.columns)

    man = users.add_prefix("m_").reindex(it.male_user_id).reset_index(drop=True)
    wom = users.add_prefix("w_").reindex(it.female_user_id).reset_index(drop=True)
    df = pd.concat([it.reset_index(drop=True), man, wom], axis=1)

    # pair (compatibility) features
    df["age_diff"] = df.m_age - df.w_age                      # man older -> positive
    df["height_diff"] = df.m_height - df.w_height
    df["attract_diff"] = df.m_attract - df.w_attract
    df["attract_sum"] = df.m_attract + df.w_attract
    for c in SIMILARITY:
        df[f"absdiff_{c}"] = (df[f"m_{c}"] - df[f"w_{c}"]).abs()
    df["kids_conflict"] = ((df.m_kids_wanted == 0) != (df.w_kids_wanted == 0)).astype(float)
    df.loc[df.m_kids_wanted.isna() | df.w_kids_wanted.isna(), "kids_conflict"] = np.nan

    # event context (known before the event: who is registered)
    size = it.groupby("event_id").agg(n_men=("male_user_id", "nunique"),
                                      n_women=("female_user_id", "nunique"))
    df = df.join(size, on="event_id")
    e = ev.reindex(df.event_id).reset_index(drop=True)
    df["m_age_in_bracket"] = (df.m_age - e.min_age_men.values) / (e.max_age_men - e.min_age_men).values
    df["w_age_in_bracket"] = (df.w_age - e.min_age_women.values) / (e.max_age_women - e.min_age_women).values

    df["y_mutual"] = df.mutual_interest.astype(int)
    df["y_m_likes"] = df.does_male_like_female.astype(int)
    df["y_w_likes"] = df.does_female_like_male.astype(int)
    df = df.drop(columns=["mutual_interest", "does_male_like_female", "does_female_like_male"])
    return df


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    man = [c for c in df.columns if c.startswith("m_")]
    wom = [c for c in df.columns if c.startswith("w_")]
    skip = {"event_id", "male_user_id", "female_user_id", "y_mutual", "y_m_likes", "y_w_likes"}
    pair = [c for c in df.columns if c not in skip and c not in man and c not in wom]
    return {"man": man, "woman": wom, "pair": pair}


if __name__ == "__main__":
    PROCESSED.mkdir(exist_ok=True)
    d = build_pairs()
    d.to_csv(PROCESSED / "pairs.csv", index=False)
    g = feature_groups(d)
    print(d.shape, {k: len(v) for k, v in g.items()})
    print(d[["y_mutual", "y_m_likes", "y_w_likes"]].mean())
