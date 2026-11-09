"""Source-specific detection filters and expression-independent context groups."""

import re
from collections.abc import Mapping
from functools import lru_cache

from .biosample_catalog import biosample_catalog, normalized_id, normalized_label
from .configuration import parameter

DEFAULT_EXPRESSION_FOLD_THRESHOLD = parameter("expression.fold_threshold")
EXPRESSION_METHOD = "context_hpa_style_linear_median_v2"
HPA_METHOD_URL = "https://www.proteinatlas.org/humanproteome/tissue/tissue%2Bspecific"

# A new source/unit never inherits another assay's detection limit. Applying
# published RNA abundance filters to returned medians is an explicit adaptation,
# not a replication of donor-level QC or HPA's published gene classifications.
DEFAULT_DETECTION_RULES = (
    {
        "source": "gtex",
        "datatype": "bulk rna-seq",
        "unit": "TPM",
        "minimum": parameter("expression.detection_gtex_tpm"),
        "basis": f'TissueEnrich TPM >= {parameter("expression.detection_gtex_tpm"):g} filter adapted to returned GTEx medians',
        "reference": "https://tissueenrich.gdcb.iastate.edu/",
    },
    {
        "source": "DICE",
        "datatype": "bulk rna-seq",
        "unit": "TPM",
        "minimum": parameter("expression.detection_dice_tpm"),
        "basis": f'DICE {parameter("expression.detection_dice_tpm"):g} TPM abundance filter adapted from mean to returned median',
        "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6289654/",
    },
    {
        "source": "HPA",
        "datatype": "bulk rna-seq",
        "unit": "nTPM",
        "minimum": parameter("expression.detection_hpa_ntpm"),
        "basis": "HPA nTPM detection limit",
        "reference": "https://www.proteinatlas.org/humanproteome/tissue/method/transcriptomics",
    },
)


def context_matches(term, observed):
    expected = " ".join(str(term).casefold().replace("-", " ").split())
    actual = " ".join(str(observed).casefold().replace("-", " ").split())
    forms = {expected}
    if expected.endswith("s") and not expected.endswith("pancreas"):
        forms.add(expected[:-1])
    return any(
        form and re.search(r"(?<!\w)" + re.escape(form) + r"s?(?!\w)", actual) for form in forms
    )


@lru_cache(maxsize=8192)
def _resolved_identity(value, axis):
    result = biosample_catalog().resolve(value, axis=axis)
    identity = result["matches"][0]["id"] if result["status"] == "resolved" else None
    return identity, result["status"], result["basis"]


def _same_concept(left, right, axis):
    catalog_axis = "cell_types" if axis == "cell_type" else "tissues"
    left_id = _resolved_identity(left, catalog_axis)[0]
    right_id = _resolved_identity(right, catalog_axis)[0]
    return left_id == right_id if left_id and right_id else context_matches(left, right)


def _selector(value):
    if not isinstance(value, Mapping) or set(value) - {"tissue", "cell_type"}:
        raise ValueError("expression context selectors require tissue and/or cell_type")
    result = {}
    for key, terms in sorted(value.items()):
        terms = (terms,) if isinstance(terms, str) else terms
        if (
            not isinstance(terms, (tuple, list))
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError("expression context selector terms must be nonempty strings")
        result[key] = sorted({term.strip() for term in terms})
    if not result:
        raise ValueError("empty expression context selector")
    return result


def expression_context(relevant_terms, *, selectors=None, context_groups=None):
    """Freeze selectors and named groups before reading gene expression.

    Organ-qualified cell phrases require both observed organ and cell type.
    Generic declared cell types have no inferred organ restriction.
    """
    terms = sorted({str(t).strip() for t in relevant_terms if str(t).strip()})
    if selectors is None:
        selectors = []
        for term in terms:
            identity = _resolved_identity(term, None)[0]
            native_axis = biosample_catalog().axis(identity) if identity else None
            cell = native_axis == "cell_types" if native_axis is not None else bool(
                re.search(
                    r"cells?\b|lymphocyte|monocyte|macrophage|eosinophil|neutrophil",
                    term,
                    re.IGNORECASE,
                )
            )
            selector = {"cell_type" if cell else "tissue": term}
            if cell and re.search(r"\b(airway|alveolar)\b", term, re.IGNORECASE):
                selector["tissue"] = ["lung", "bronchus", "airway epithelium"]
            elif cell and re.search(r"\bpancreatic\b", term, re.IGNORECASE):
                selector["tissue"] = ["pancreas", "pancreatic islet"]
            selectors.append(selector)
    selected = sorted({repr(_selector(s)): _selector(s) for s in selectors}.values(), key=repr)
    if context_groups is None:
        # Biological group definitions are fixed, never assembled by choosing
        # the highest-expressing tissues for an individual gene.
        catalog = (
            (
                "respiratory_tissues",
                "tissue",
                ("lung", "bronchus"),
                "Respiratory organ and conducting airway",
            ),
            (
                "adaptive_lymphocytes",
                "cell_type",
                ("T cell", "B cell"),
                "Adaptive lymphocyte lineages",
            ),
            (
                "type2_immune_effectors",
                "cell_type",
                ("mast cell", "eosinophil"),
                "Type 2 immune effector cell types",
            ),
        )
        context_groups = [
            {"id": name, "members": [{axis: term} for term in members], "rationale": rationale}
            for name, axis, members, rationale in catalog
            if all(
                any(
                    set(s) == {axis} and any(_same_concept(t, member, axis) for t in s[axis])
                    for s in selected
                )
                for member in members
            )
        ]
        # Preserve the caller's literal terms (e.g. plural cell names).
        for group in context_groups:
            group["members"] = [
                next(
                    s
                    for s in selected
                    if set(s) == set(m)
                    and any(
                        _same_concept(t, next(iter(m.values())), next(iter(m)))
                        for t in next(iter(s.values()))
                    )
                )
                for m in group["members"]
            ]
    groups, ids = [], set()
    for group in context_groups:
        if not isinstance(group, Mapping):
            raise TypeError("invalid expression context group")
        name, rationale = group.get("id"), group.get("rationale")
        members = sorted([_selector(s) for s in group.get("members", ())], key=repr)
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in ids
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            raise ValueError("context groups require a unique id and biological rationale")
        if not 2 <= len(members) <= 5 or len({repr(m) for m in members}) != len(members):
            raise ValueError("context groups require 2-5 distinct predefined members")
        if any(member not in selected for member in members):
            raise ValueError("every expression group member must be declared in context selectors")
        axes = {"cell_type" if "cell_type" in m else "tissue" for m in members}
        if len(axes) != 1:
            raise ValueError("context groups cannot mix tissues with cell types")
        ids.add(name)
        groups.append({"id": name, "members": members, "axis": axes.pop(), "rationale": rationale})
    return {
        "terms": terms,
        "selectors": selected,
        "predefined_groups": sorted(groups, key=lambda group: group["id"]),
        "groups_version": "anatomical_and_cell_groups_v2",
        "group_aggregation": "equal_weight_declared_members_mean_of_all_resolved_descendants",
        "group_method_interpretation": "custom_declared_lineage_summary_not_official_hpa_annotation",
        "missingness_policy": "available_values_descriptive_only_until_reference_and_members_complete",
    }


def expression_selectors_from_axes(*, organs=(), tissues=(), cell_types=()):
    """Build separate tissue and cell selectors from a frozen OT context.

    OT context planning has three biological levels.  Expression comparison has
    two measurement axes: organ/tissue expression and cell-type expression.
    Organs are therefore selectors on the tissue axis, while cell types are
    selectors on the cell axis.  No organ is inferred for a cell selector here;
    an organ-qualified cell relationship must be explicitly present in the
    frozen context or the source row.
    """
    selectors = [
        {"tissue": str(term).strip()}
        for term in (*organs, *tissues)
        if isinstance(term, str) and term.strip()
    ]
    selectors.extend(
        {"cell_type": str(term).strip()}
        for term in cell_types
        if isinstance(term, str) and term.strip()
    )
    return tuple(
        sorted(
            {repr(_selector(selector)): _selector(selector) for selector in selectors}.values(),
            key=repr,
        )
    )


@lru_cache(maxsize=32768)
def _dimension_match(terms, identifier, label, axis, selected_ids, descendants, strict):
    catalog = biosample_catalog()
    explicit, _, id_basis = _resolved_identity(identifier, axis)
    if id_basis != "catalog_id":
        explicit = None
    labelled, label_status, _ = _resolved_identity(label, axis)
    # Preserve a known ID's optional condition label, but reject a second known
    # identity or an unrecognized ontology ID. A lung ID cannot mean liver.
    if id_basis == "unrecognized_id" or (explicit and labelled and explicit != labelled):
        return False
    if explicit and label_status == "ambiguous":
        candidates = catalog.resolve(label, axis=axis)["matches"]
        if explicit not in {item["id"] for item in candidates}:
            return False
    observed = explicit or labelled
    targets = selected_ids if strict else tuple(
        identity for term in terms if (identity := _resolved_identity(term, axis)[0])
    )
    if observed and any(catalog.relation(observed, target, allow_descendants=descendants)
                        for target in targets):
        return True
    if strict or explicit or id_basis == "unrecognized_id":
        return False
    # Historical adapters and explicitly supplied custom contexts can contain
    # non-ontology labels. This compatibility path is never used with a frozen
    # OT mapping, and cannot override contradictory resolved identities.
    return any(
        context_matches(term, value)
        for term in terms
        if not observed or not _resolved_identity(term, axis)[0]
        for value in (identifier, label) if value
    )


def matches_selector(selector, sample, *, context_mapping=None):
    """Match each declared axis with the same frozen OT ID/ancestor policy as QTLs."""
    catalog = biosample_catalog()
    strict = context_mapping is not None
    if strict and (not isinstance(context_mapping, Mapping) or
                   not catalog.recorded_with(context_mapping.get("catalog", {}).get("catalog_sha256"))):
        return False
    descendants = (context_mapping.get("allow_catalog_descendants") is True if strict
                   else parameter("context.allow_catalog_descendants"))
    dimensions = {"tissue": (sample[0], sample[2]), "cell_type": (sample[1], sample[3])}
    for key, terms in selector.items():
        axis = "cell_types" if key == "cell_type" else "tissues"
        targets = set()
        if strict:
            for concept in context_mapping.get("concepts", ()):
                if concept.get("status") != "resolved" or (
                    concept.get("axis") == "cell_types") != (key == "cell_type"):
                    continue
                for match in concept.get("matches", ()):
                    if any(normalized_id(term) == match["id"] or
                           normalized_label(term) in {
                               normalized_label(concept.get("input", "")),
                               normalized_label(match.get("label", "")),
                           } or _resolved_identity(term, axis)[0] == match["id"]
                           for term in terms):
                        targets.add(match["id"])
        if not _dimension_match(tuple(terms), *dimensions[key], axis,
                                tuple(sorted(targets)), descendants, strict):
            return False
    return True


def resolved_sample_identity(sample, axis):
    """Return an unambiguous identity for grouping; never merge condition labels."""
    index = 1 if axis == "cell_type" else 0
    catalog_axis = "cell_types" if axis == "cell_type" else "tissues"
    identity, _, basis = _resolved_identity(sample[index], catalog_axis)
    if basis != "catalog_id":
        identity = None
    labelled = _resolved_identity(sample[index + 2], catalog_axis)[0]
    if basis == "unrecognized_id" or (identity and labelled and identity != labelled):
        return None
    return identity or labelled
