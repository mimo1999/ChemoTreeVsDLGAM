"""Shared parsing for the VMD (Value / Missingness / Delta) feature-name convention.

ml_feature_matrix_builder.py emits VMD-family columns in one of two shapes,
depending on --feature_method:

    concatenate (default):  <channel>_<itemid>_<bin>   e.g. "X_50801_5"
    aggregate:               <channel>_<itemid>          e.g. "X_50801"
                             (the daily bin has already been averaged away)

<channel> is one of X (value), M (missingness flag), or D (delta / time
since last observation); <itemid> is always numeric (a MIMIC/UKEr lab
itemid). Every wrapper that needs to recover (channel, itemid[, bin]) from a
column name should call `parse_vmd_column` below instead of hand-rolling its
own regex, so the convention only needs to change in one place. Wrappers
that need more than the raw tuple (e.g. "give me just the itemid, I don't
care about bin or channel") should define their own small function on top of
it rather than matching column names themselves -- see
`he_ebm_wrapper._expert_itemid` for an example.
"""

import re

_VMD_PATTERN = re.compile(r"^([XMD])_(\d+)(?:_(\d+))?$", re.IGNORECASE)


def parse_vmd_column(name) -> "tuple[str, int, int | None] | None":
    """Match a column name against the VMD convention.

    Returns (channel, itemid, bin): channel is one of 'X', 'M', 'D'
    (uppercased); itemid is an int; bin is an int, or None for a
    feature_method="aggregate" column (no per-day bin). Returns None if
    `name` doesn't follow the VMD convention at all (a static demographic
    column, or an already-aggregated feat_type="A" column).
    """
    m = _VMD_PATTERN.match(str(name))
    if not m:
        return None
    channel, itemid, bin_id = m.groups()
    return channel.upper(), int(itemid), (int(bin_id) if bin_id is not None else None)


def san(name) -> str:
    """Sanitise a feature name into a safe R identifier fragment."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(name))
