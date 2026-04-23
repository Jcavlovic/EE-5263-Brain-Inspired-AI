"""Derive gender labels for LFW identities from their first names."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import gender_guesser.detector as gd


MALE_SET = {"male", "mostly_male"}
FEMALE_SET = {"female", "mostly_female"}


@dataclass
class Sample:
    path: str
    label: int  # 0 = male, 1 = female
    person: str


def build_labeled_samples(lfw_root: str | os.PathLike) -> list[Sample]:
    """Walk the LFW directory and build (image path, gender label) tuples.

    Identities whose first name maps to `andy` (androgynous) or `unknown`
    are skipped because their label is ambiguous.
    """
    root = Path(lfw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"LFW root not found: {root}")

    detector = gd.Detector(case_sensitive=False)
    samples: list[Sample] = []

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        first = person_dir.name.split("_", 1)[0]
        guess = detector.get_gender(first)
        if guess in MALE_SET:
            label = 0
        elif guess in FEMALE_SET:
            label = 1
        else:
            continue
        for img in sorted(person_dir.glob("*.jpg")):
            samples.append(Sample(str(img), label, person_dir.name))

    return samples


def summarize(samples: list[Sample]) -> dict[str, int]:
    male = sum(1 for s in samples if s.label == 0)
    female = sum(1 for s in samples if s.label == 1)
    return {"male": male, "female": female, "total": male + female}
