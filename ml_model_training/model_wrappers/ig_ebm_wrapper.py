"""
IG-EBM: interaction-graph-forced EBM.

Reads the empirical interaction graph produced upstream for a cohort
(`MIMIC_IV/saved_data/interactions/top_interactions.csv` -- one row per
(cohort, fold, itemid-pair) with a `feature_importance`-derived `importance`
score for that pair), builds a single generalized ranking of ALL itemid-pairs
across every fold of the cohort, filters that full ranking down to the
edges that are actually feasible given the features that survived this
fold's upstream selection, keeps the top `top_n_interactions` of that
feasible subset, maps each onto the actual VMD feature columns present at
fit time, and forces exactly those column-pairs as EBM interaction terms
instead of letting EBM's own FAST auto-search pick them.

Ranking happens over the full graph before feasibility is checked, and
feasibility is checked before the top_n cut -- in that order. Cutting to
top_n first and checking feasibility second would silently shrink the
forced-interaction count below top_n any time an infeasible edge (an
itemid upstream feature selection dropped) happened to land inside the
top-n window of the global ranking.

Ranking methodology
--------------------
A pair is only worth forcing if it was *consistently* important, not just
important in one lucky fold. Each edge (node1, node2) is scored across the
cohort's folds by:

    cooccurrence  = number of folds where importance > 0
    mean_importance = mean importance across all folds (0 counts for
                      folds where the pair had no measured importance)

Edges are ranked by (cooccurrence desc, mean_importance desc) and the top
`top_n_interactions` are kept. This directly implements "generalized
ranking across folds using feature_importance and cooccurrence across
folds" -- an edge that shows up strong in every fold outranks one that
spikes once and vanishes.

Column mapping
---------------
The interaction graph is defined over itemids, but VMD columns are per-day
-bin (`X_<itemid>_<bin>`, feature_method="concatenate") or already
aggregated (`X_<itemid>`, feature_method="aggregate"). For a selected
(itemid1, itemid2) edge, every same-bin column pair is forced (bin_i ==
bin_j, including the aggregate case where both bins are None) -- i.e. "how
lab A and lab B interact on the same day", which is the only reading of a
cross-lab edge that doesn't also require guessing a lag structure. An edge
is silently dropped if either itemid has no surviving column in the
current fold's feature set (upstream feature selection may have removed
it).
"""

from collections.abc import Iterable
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config.constants import RANDOM_SEED, PROJECT_ROOT, BEST_PARAMS
from utils.feature_naming import parse_vmd_column
from .ebm_wrapper import EBMClassifier


# ── Interaction graph loading & ranking ────────────────────────────────────

def _default_interactions_path() -> Path:
    return Path(PROJECT_ROOT) / "MIMIC_IV" / "saved_data" / "interactions" / "top_interactions.csv"


@lru_cache(maxsize=8)
def _load_interaction_graph(interactions_path: str) -> pd.DataFrame:
    """Parse top_interactions.csv, cached per path (static across folds)."""
    return pd.read_csv(interactions_path)


def rank_interactions_across_folds(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """
    Collapse a cohort's per-fold edge importances into one generalized
    ranking, ordered best-first by (cooccurrence, mean_importance).

    cooccurrence : number of folds in which the edge had importance > 0.
    mean_importance : mean importance across ALL of the cohort's folds
        (folds where the edge is absent or zero count as 0, so a pair that
        only appears in 1 of 5 folds can't outrank one seen in all 5 just
        by having a single huge spike).
    """
    cohort_df = df[df["cohort"] == cohort]
    if cohort_df.empty:
        raise ValueError(f"[IG-EBM] No rows for cohort={cohort!r} in interaction graph CSV")

    n_folds = cohort_df["fold"].nunique()
    ranked = (
        cohort_df.groupby(["node1", "node2"], as_index=False)
        .agg(
            cooccurrence=("importance", lambda s: int((s > 0).sum())),
            total_importance=("importance", "sum"),
        )
    )
    ranked["mean_importance"] = ranked["total_importance"] / n_folds
    ranked = ranked.sort_values(
        ["cooccurrence", "mean_importance"], ascending=[False, False]
    ).reset_index(drop=True)
    return ranked[["node1", "node2", "cooccurrence", "mean_importance"]]


def build_interaction_pairs_from_graph(
    feature_names: Iterable[str],
    ranked_edges: pd.DataFrame,
    top_n: int,
) -> tuple[list[tuple[str, str]], int, int]:
    """
    Filter the FULL ranked (node1, node2) itemid-pair list down to those
    that are actually feasible given `feature_names` (both itemids have at
    least one surviving column), THEN take the top `top_n` of that feasible
    subset and map each onto same-bin column-pairs.

    Ranking first and slicing to top_n before checking feasibility would
    silently shrink the forced-interaction count below top_n whenever an
    infeasible edge (itemid dropped by upstream feature selection) lands
    inside the top-n cut. Filtering to feasible edges first means the
    result contains up to top_n edges whenever that many feasible edges
    exist at all, regardless of where infeasible edges fall in the global
    ranking.

    Returns
    -------
    pairs : list[tuple[str, str]]
        Same-bin column-pairs for the top `top_n` feasible edges.
    n_feasible : int
        Total number of edges (across the whole ranking) that were
        feasible, before the top_n cut.
    n_total : int
        Total number of ranked edges considered (feasible + infeasible).
    """
    by_itemid: dict[int, list[tuple[str, "int | None"]]] = defaultdict(list)
    for name in feature_names:
        parsed = parse_vmd_column(name)
        if parsed is not None and parsed[0] == "X":
            by_itemid[parsed[1]].append((name, parsed[2]))

    feasible_mask = ranked_edges.apply(
        lambda row: int(row["node1"]) in by_itemid and int(row["node2"]) in by_itemid,
        axis=1,
    )
    feasible_edges = ranked_edges[feasible_mask]

    pairs: list[tuple[str, str]] = []
    for _, row in feasible_edges.head(top_n).iterrows():
        cols1 = by_itemid[int(row["node1"])]
        cols2 = by_itemid[int(row["node2"])]
        for name1, bin1 in cols1:
            for name2, bin2 in cols2:
                if bin1 == bin2:
                    pairs.append((name1, name2))
    return pairs, len(feasible_edges), len(ranked_edges)


class IGEBMClassifier(EBMClassifier):
    """EBM whose interaction terms are forced from the empirical interaction
    graph (top_interactions.csv) instead of EBM's own FAST auto-search. See
    module docstring for the ranking and column-mapping methodology.

    Parameters
    ----------
    cohort : str
        Selects both BEST_PARAMS['EBM'][cohort] (the tuned EBM backbone --
        same as the production flat EBM) and the `cohort` column of the
        interaction graph CSV. Required: interaction graph rows from one
        cohort are not meaningful for another, and there is no fallback
        hyperparameter set (mirrors HE-EBM's `cohort` requirement).
    top_n_interactions : int, default 4
        Number of top-ranked itemid-pair edges (by cooccurrence, then mean
        importance, across the cohort's folds) to force as interactions.
    interactions_csv_path : str | None
        Path to top_interactions.csv. Auto-detected from PROJECT_ROOT if
        None.
    max_bins, outer_bags, max_rounds, learning_rate, smoothing_rounds,
    interaction_smoothing_rounds, early_stopping_rounds,
    early_stopping_tolerance, inner_bags
        Fallback EBM hyperparameters used only when no BEST_PARAMS['EBM']
        entry exists for `cohort`.
    random_state : int, default RANDOM_SEED

    Note: `interactions` is NOT a constructor parameter here -- it is
    computed inside `fit()` from the interaction graph and whatever
    `feature_names` are actually passed in (post upstream feature
    selection), so the forced-pair set always matches the columns present
    at fit time.
    """

    def __init__(
        self,
        cohort: str | None = None,
        top_n_interactions: int = 4,
        interactions_csv_path: str | None = None,
        max_bins: int = 128,
        outer_bags: int = 4,
        max_rounds: int = 100,
        learning_rate: float = 0.1,
        smoothing_rounds: int = 25,
        interaction_smoothing_rounds: int = 25,
        early_stopping_rounds: int = 30,
        early_stopping_tolerance: float = 1e-4,
        inner_bags: int = 0,
        random_state: int = RANDOM_SEED,
    ):
        self.cohort = cohort
        self.top_n_interactions = top_n_interactions
        self.interactions_csv_path = interactions_csv_path
        super().__init__(
            max_bins=max_bins,
            interactions=[],  # placeholder; fit() overrides with graph-derived pairs
            outer_bags=outer_bags,
            max_rounds=max_rounds,
            learning_rate=learning_rate,
            smoothing_rounds=smoothing_rounds,
            interaction_smoothing_rounds=interaction_smoothing_rounds,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            inner_bags=inner_bags,
            random_state=random_state,
        )

    def _resolve_backbone_params(self) -> dict:
        """EBM backbone hyperparameters: BEST_PARAMS['EBM'][cohort] if
        available (matches the production flat EBM exactly, isolating
        `interactions` as the only variable under test), else this
        wrapper's own fallback attributes."""
        if self.cohort:
            best = BEST_PARAMS.get("EBM", {}).get(self.cohort)
            if best:
                print(f"[IG-EBM] Using best EBM params for cohort '{self.cohort}'")
                return dict(best)
        print("[IG-EBM] No best EBM params found for cohort — using wrapper defaults")
        return dict(
            max_bins=self.max_bins,
            outer_bags=self.outer_bags,
            max_rounds=self.max_rounds,
            learning_rate=self.learning_rate,
            smoothing_rounds=self.smoothing_rounds,
            interaction_smoothing_rounds=self.interaction_smoothing_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            early_stopping_tolerance=self.early_stopping_tolerance,
        )

    def fit(self, X, y, feature_names: Iterable[str] | None = None, sample_weight=None):
        if not self.cohort:
            raise ValueError(
                "[IG-EBM] `cohort` is required -- interaction-graph rows from one "
                "cohort are not meaningful for another."
            )

        if feature_names is None:
            feature_names = list(X.columns)
        feature_names = list(feature_names)

        interactions_path = self.interactions_csv_path or _default_interactions_path()
        graph_df = _load_interaction_graph(str(interactions_path))
        ranked_edges = rank_interactions_across_folds(graph_df, self.cohort)
        pairs, n_feasible, n_total = build_interaction_pairs_from_graph(
            feature_names, ranked_edges, self.top_n_interactions
        )

        print(
            f"[IG-EBM] cohort='{self.cohort}': {n_total} candidate edges -> "
            f"{n_feasible} feasible given {len(feature_names)} available features -> "
            f"top {min(self.top_n_interactions, n_feasible)} by (cooccurrence, "
            f"mean_importance) -> {len(pairs)} same-bin column-pairs forced"
        )
        if n_feasible < self.top_n_interactions:
            print(
                f"[IG-EBM] NOTE: only {n_feasible} of {n_total} ranked edges were "
                f"feasible (both itemids survived feature selection) -- fewer than "
                f"the requested top_n_interactions={self.top_n_interactions}."
            )
        if not pairs:
            print(
                "[IG-EBM] WARNING: no forced interaction pairs survived column mapping "
                "-- no ranked edge had both itemids present in the available features. "
                "Falling back to a plain-main-effects EBM (no interactions)."
            )

        backbone = self._resolve_backbone_params()
        self.max_bins = backbone["max_bins"]
        self.outer_bags = backbone["outer_bags"]
        self.max_rounds = backbone["max_rounds"]
        self.learning_rate = backbone["learning_rate"]
        self.smoothing_rounds = backbone["smoothing_rounds"]
        self.interaction_smoothing_rounds = backbone["interaction_smoothing_rounds"]
        self.early_stopping_rounds = backbone["early_stopping_rounds"]
        self.early_stopping_tolerance = backbone["early_stopping_tolerance"]
        self.interactions = pairs

        return super().fit(X, y, feature_names=feature_names, sample_weight=sample_weight)
