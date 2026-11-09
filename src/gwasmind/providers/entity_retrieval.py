"""Retrieve shared study-locus objects once, then bind exact variant members.

Only the retrieval representation changes. Each compatibility view still carries
its own measured member P/beta/SE/PP and the original source/proxy binding.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from ..audit import sha256_json
from ..configuration import parameter
from ..expression_policy import context_matches
from ..models import EvidenceKind
from ..molecular_context import MOLECULAR_RETRIEVAL_SCOPE, molecular_context_match
from ..qtl import native_qtl_study_type
from ..supplementary_molecular import molecular_context_class
from .retrieval_recovery import (
    ANALYSIS_VARIANTS,
    ENTITY_RECOVERY_ACTIVE,
    failure_category,
    recovery_progress,
    safe_transport_failure,
)

CORE_FIELDS = (
    "id: studyLocusId studyLocusId studyId studyType chromosome position beta "
    "pValueMantissa pValueExponent finemappingMethod confidence qualityControls qtlGeneId variant { id }"
)
STUDY_FIELDS = (
    "id studyType traitFromSource pubmedId publicationDate publicationTitle "
    "publicationFirstAuthor publicationJournal target { id approvedSymbol } "
    "biosample { biosampleId biosampleName }"
)
STUDY_DIRECTORY_FIELDS = (
    "id studyType traitFromSource target { id approvedSymbol } "
    "biosample { biosampleId biosampleName }"
)
MEMBER_FIELDS = (
    "variant { id } posteriorProbability logBF is95CredibleSet is99CredibleSet "
    "pValueMantissa pValueExponent beta standardError r2Overall"
)
L2G_FIELDS = "score target { id approvedSymbol }"
COLOC_FIELDS = (
    "rightStudyType chromosome colocalisationMethod numberColocalisingVariants h3 h4 clpp "
    "betaRatioSignAverage otherStudyLocus { studyLocusId studyId studyType qtlGeneId }"
)


@dataclass
class EntityVariantResult:
    variants: list
    coverage: list


class EntityEvidenceLoader:
    def __init__(self, provider):
        self.provider = provider
        # These maps are the working evidence graph, not copies per variant.
        self.cores, self.studies, self.memberships, self.connections = {}, {}, {}, {}
        self.failures = {}
        self.initial_pages = {}
        self.connection_prefetch_attempted = set()
        self.study_directory = {}

    def stats(self):
        return {**self.provider.request_cache.summary(), "unique_credible_sets": len(self.cores),
                "unique_studies": len(self.studies), "unique_connections": len(self.connections)}

    async def _query(self, query, variables):
        from .live import _unwrap_query_result
        response = await self.provider._call_tool("query_open_targets_graphql", {
            "query_string": query, "variables": variables,
        })
        return _unwrap_query_result(response)

    async def _entities(self, identifiers, *, root, argument, fields, cache, common_variables=None, batch_size=None,
                        _recovery_queue=None):
        from .live import LiveProviderError, _graphql_error_summary, _requested_entity
        size = (parameter("retrieval.study_query_batch_size" if root == "study" and fields == STUDY_FIELDS
                          and not common_variables else "retrieval.entity_query_batch_size")
                if batch_size is None else batch_size)
        if type(size) is not int or size < 1:
            raise ValueError("Entity request batch size must be a positive integer")
        recovery_owner = _recovery_queue is None
        pending_recovery = [] if recovery_owner else _recovery_queue
        missing = sorted(set(identifiers) - set(cache) - {key[1] for key in self.failures if key[0] == root})
        async def group(chunk):
            declarations = ", ".join(f"$entity{i}: String!" for i in range(len(chunk)))
            if common_variables:
                declarations += ", $memberVariants: [String!]!"
            selections = " ".join(f"entity{i}: {root}({argument}: $entity{i}) {{ {fields} }}" for i in range(len(chunk)))
            try:
                data = await self._query(f"query GWASMindSharedEntities({declarations}) {{ {selections} }}",
                                         {**(common_variables or {}), **{f"entity{i}": value for i, value in enumerate(chunk)}})
                retry_dense = []
                for i, identifier in enumerate(chunk):
                    try:
                        alias_errors = (data.get("__gwasmind_alias_errors") or {}).get(f"entity{i}")
                        if alias_errors:
                            exc = LiveProviderError(_graphql_error_summary(alias_errors))
                            if failure_category(exc) in {"query_too_expensive", "response_size_limit"}:
                                if len(chunk) > 1 or root == "credibleSet" and common_variables:
                                    retry_dense.append(identifier)
                                    continue
                            raise exc
                        value = _requested_entity(data, f"entity{i}", identifier)
                        if value is None:
                            raise LiveProviderError("Referenced entity metadata was not returned")
                        if root == "credibleSet" and value.get("studyLocusId") != identifier:
                            raise LiveProviderError("Credible-set metadata has contradictory identity")
                        cache[identifier] = value
                    except LiveProviderError as exc:
                        self.failures[(root, identifier)] = failure_category(exc)
                if retry_dense:
                    if root == "credibleSet" and common_variables:
                        await self._entities(retry_dense, root=root, argument=argument,
                                             fields=CORE_FIELDS, cache=cache, batch_size=size,
                                             _recovery_queue=pending_recovery)
                    else:
                        # An alias-level cost error must shrink too. Retrying the
                        # same singleton can otherwise recurse through its cached
                        # error forever; successful siblings must never repeat.
                        smaller = max(1, len(chunk) // 2)
                        recovery_progress("entities", "splitting", unit="entities", total_count=len(retry_dense),
                                          completed_count=0, reason="query_too_expensive")
                        for offset in range(0, len(retry_dense), smaller):
                            await group(retry_dense[offset:offset + smaller])
            except LiveProviderError as exc:
                if failure_category(exc) in {"query_too_expensive", "response_size_limit"} and len(chunk) > 1:
                    middle = len(chunk) // 2
                    recovery_progress("entities", "splitting", unit="entities", total_count=len(chunk),
                                      completed_count=0, reason="query_too_expensive")
                    await group(chunk[:middle]); await group(chunk[middle:])
                    return
                if (failure_category(exc) in {"query_too_expensive", "response_size_limit"}
                        and root == "credibleSet" and common_variables):
                    # A single dense entity may still exceed the server's cost
                    # limit. Retrieve its core first; each connection then uses
                    # the same bounded, independently shrinkable page reader.
                    await self._entities(chunk, root=root, argument=argument, fields=CORE_FIELDS,
                                         cache=cache, batch_size=size, _recovery_queue=pending_recovery)
                    return
                for identifier in chunk:
                    self.failures[(root, identifier)] = failure_category(exc)
                if (not ENTITY_RECOVERY_ACTIVE.get() and not getattr(exc, "entity_recovery_exhausted", False)
                        and safe_transport_failure(exc).get("retryable") is True):
                    pending_recovery.append((group, tuple(chunk), cache, root))
            recovery_progress("entities", "running", unit="entities", total_count=len(identifiers),
                              completed_count=sum(identifier in cache for identifier in identifiers), **self.stats())
        for offset in range(0, len(missing), size):
            await group(missing[offset:offset + size])
        if (recovery_owner and pending_recovery and not ENTITY_RECOVERY_ACTIVE.get()
                and parameter("retrieval.entity_recovery_attempts") > 0):
            pending_ids = {(kind, identifier): completed for _call, ids, completed, kind in pending_recovery
                           for identifier in ids if identifier not in completed}
            if not pending_ids:
                return
            recovery_progress("entities", "retrying", unit="entities", attempt=2,
                              completed_count=0, total_count=len(pending_ids), **self.stats())
            token = ENTITY_RECOVERY_ACTIVE.set(True)
            try:
                # Keep ``identifiers`` bound to this request: ``group`` reports progress from it.
                for call, chunk_ids, completed, kind in pending_recovery:
                    remaining = [identifier for identifier in chunk_ids if identifier not in completed]
                    if not remaining:
                        continue
                    for identifier in remaining:
                        self.failures.pop((kind, identifier), None)
                    await call(remaining)
            finally:
                ENTITY_RECOVERY_ACTIVE.reset(token)
            unresolved = sum(identifier not in completed for (_kind, identifier), completed in pending_ids.items())
            recovery_progress("entities", "completed" if not unresolved else "incomplete", unit="entities", attempt=2,
                              completed_count=len(pending_ids)-unresolved, total_count=len(pending_ids), **self.stats())

    async def discover(self, variant_ids):
        """The compound root selects distinct study-locus objects for this set."""
        from .live import LiveProviderError
        query = (
            "query GWASMindSharedLoci($variantIds: [String!]!, $index: Int!, $size: Int!) { "
            "credibleSets(variantIds: $variantIds, page: {index: $index, size: $size}) { "
            "count rows { studyLocusId } } }"
        )
        size = parameter("retrieval.credible_set_page_size")
        identifiers, seen, count = [], set(), None
        reason = None
        budget = parameter("retrieval.credible_set_max_continuation_requests") + 1
        for _ in range(budget):
            try:
                data = await self._query(query, {"variantIds": sorted(set(variant_ids)), "index": len(identifiers) // size, "size": size})
                connection = data.get("credibleSets")
                declared, rows = self._page(connection, expected=count, size=size, offset=len(identifiers))
                page_ids = [row.get("studyLocusId") for row in rows]
                if any(not isinstance(value, str) or not value for value in page_ids) or len(set(page_ids)) != len(page_ids) or seen.intersection(page_ids):
                    raise LiveProviderError("Credible-set discovery returned invalid or duplicate identities")
                identifiers.extend(page_ids); seen.update(page_ids); count = declared
                recovery_progress("credible_sets", "running", unit="entities", completed_count=len(identifiers),
                                  total_count=count, **self.stats())
                if len(identifiers) == count:
                    return identifiers, None
            except LiveProviderError as exc:
                reason = failure_category(exc)
                if reason in {"query_too_expensive", "response_size_limit"} and size > 1:
                    size = max(value for value in range(1, size // 2 + 1) if not len(identifiers) % value)
                    continue
                break
        return identifiers, reason or "continuation_budget_exhausted"

    @staticmethod
    def _page(connection, *, expected, size, offset):
        from .live import LiveProviderError
        if not isinstance(connection, dict):
            raise LiveProviderError("Shared evidence connection must be an object")
        count, rows = connection.get("count"), connection.get("rows")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                or expected is not None and count != expected or not isinstance(rows, list)
                or any(not isinstance(row, dict) for row in rows)
                or len(rows) != min(size, count - offset)):
            raise LiveProviderError("Shared evidence page changed count or returned incomplete rows")
        return count, rows

    async def connection(self, identifier, field, fields, *, variant_ids=(), budget=None):
        from .live import LiveProviderError
        key = (identifier, field, tuple(variant_ids))
        target = self.memberships if field == "locus" else self.connections
        if key in target:
            return target[key]
        size = parameter("retrieval.credible_set_page_size")
        extra_definition = ", $variantIds: [String!]!" if field == "locus" else ""
        extra_argument = "variantIds: $variantIds, " if field == "locus" else ""
        query = (
            f"query GWASMindSharedConnection($studyLocusId: String!, $index: Int!, $size: Int!{extra_definition}) {{ "
            "credibleSet(studyLocusId: $studyLocusId) { id: studyLocusId "
            f"{field}({extra_argument}page: {{index: $index, size: $size}}) {{ count rows {{ {fields} }} }} }} }}"
        )
        rows, hashes, count, reason = [], set(), None, None
        member_ids = set()
        initial = self.initial_pages.get(key)
        if initial is not None:
            try:
                count, first = self._page(initial, expected=None, size=size, offset=0)
                if field == "locus":
                    identities = self._member_ids(first)
                    if len(set(identities)) != len(identities) or not set(identities).issubset(variant_ids):
                        raise LiveProviderError("Prefetched members do not match the analysis variants")
                    member_ids.update(identities)
                rows = list(first)
                hashes = {sha256_json(row) for row in first}
                if len(hashes) != len(rows):
                    raise LiveProviderError("Prefetched connection has duplicated rows")
            except LiveProviderError as exc:
                reason = failure_category(exc)
                result = {"count": 0, "rows": [], "coverage": {"complete": False, "reason": reason,
                    "declared_count": None, "retrieved_count": 0}}
                target[key] = result
                return result
        while True:
            if count is not None and len(rows) == count:
                break
            if rows and budget is not None and budget[0] <= 0:
                reason = "continuation_budget_exhausted"; break
            if rows and budget is not None:
                budget[0] -= 1
            try:
                variables = {"studyLocusId": identifier, "index": len(rows) // size, "size": size}
                if field == "locus": variables["variantIds"] = list(variant_ids)
                data = await self._query(query, variables)
                value = data.get("credibleSet")
                if not isinstance(value, dict) or value.get("id") != identifier:
                    raise LiveProviderError("Shared connection returned a different study-locus identity")
                declared, page = self._page(value.get(field), expected=count, size=size, offset=len(rows))
                page_hashes = [sha256_json(row) for row in page]
                if len(set(page_hashes)) != len(page_hashes) or hashes.intersection(page_hashes):
                    raise LiveProviderError("Shared connection returned repeated evidence rows")
                if field == "locus":
                    ids = self._member_ids(page)
                    if len(set(ids)) != len(ids) or member_ids.intersection(ids) or not set(ids).issubset(variant_ids):
                        raise LiveProviderError("Filtered members do not match the submitted analysis variants")
                    member_ids.update(ids)
                rows.extend(page); hashes.update(page_hashes); count = declared
                reason = None
                recovery_progress("colocalisation" if field == "colocalisation" else "entities", "running",
                    entity_id=identifier[:160], unit="records", completed_count=len(rows), total_count=count, **self.stats())
                if len(rows) == count:
                    break
            except LiveProviderError as exc:
                reason = failure_category(exc)
                # A rejected document can shrink at an aligned offset without
                # re-fetching successful pages or changing index semantics.
                if reason in {"query_too_expensive", "response_size_limit"} and size > 1:
                    divisors = [value for value in range(1, size) if not len(rows) % value]
                    size = max(value for value in divisors if value <= max(1, size // 2))
                    continue
                break
        result = {"count": count if count is not None else len(rows), "rows": rows,
                  "coverage": {"complete": reason is None and count is not None and len(rows) == count,
                               "declared_count": count, "retrieved_count": len(rows), "reason": reason,
                               "index_semantics": "page_number", "page_size": size}}
        target[key] = result
        return result

    @staticmethod
    def _member_ids(rows):
        from .live import LiveProviderError
        identities = []
        for row in rows:
            value = row.get("variant")
            identifier = value.get("id") if isinstance(value, dict) else None
            if not isinstance(identifier, str) or not identifier:
                raise LiveProviderError("Filtered member has no verifiable variant identity")
            identities.append(identifier)
        return identities

    async def _prefetch_l2g_pages(self, identifiers):
        """Add a missing connection capability without replacing cached cores."""
        await self._prefetch_connection_pages(identifiers, "l2GPredictions", L2G_FIELDS)

    async def _prefetch_member_pages(self, identifiers, scope):
        """Batch first membership pages for cached cores whose member scope changed."""
        await self._prefetch_connection_pages(identifiers, "locus", MEMBER_FIELDS, scope=scope)

    async def _prefetch_connection_pages(self, identifiers, field, fields, *, scope=()):
        from .live import LiveProviderError, _graphql_error_summary, _requested_entity
        scope = tuple(scope)
        target = self.memberships if field == "locus" else self.connections
        def key_for(identifier):
            return identifier, field, scope
        missing = sorted(identifier for identifier in set(identifiers) if identifier in self.cores
                         and key_for(identifier) not in self.initial_pages
                         and key_for(identifier) not in target
                         and key_for(identifier) not in self.connection_prefetch_attempted)
        page_size = parameter("retrieval.credible_set_page_size")
        member_argument = "variantIds: $memberVariants, " if field == "locus" else ""
        async def group(chunk):
            self.connection_prefetch_attempted.update(key_for(identifier) for identifier in chunk)
            declarations = ", ".join(f"$entity{i}: String!" for i in range(len(chunk)))
            if field == "locus":
                declarations += ", $memberVariants: [String!]!"
            selections = " ".join(
                f"entity{i}: credibleSet(studyLocusId: $entity{i}) {{ id: studyLocusId "
                f"{field}({member_argument}page: {{index: 0, size: {page_size}}}) {{ count rows {{ {fields} }} }} }}"
                for i in range(len(chunk)))
            variables = {f"entity{i}": identifier for i, identifier in enumerate(chunk)}
            if field == "locus":
                variables["memberVariants"] = list(scope)
            retry_dense = []
            try:
                data = await self._query(f"query GWASMindSharedConnectionPages({declarations}) {{ {selections} }}",
                                         variables)
                for i, identifier in enumerate(chunk):
                    try:
                        errors = (data.get("__gwasmind_alias_errors") or {}).get(f"entity{i}")
                        if errors: raise LiveProviderError(_graphql_error_summary(errors))
                        value = _requested_entity(data, f"entity{i}", identifier)
                        if value is None: raise LiveProviderError("Connection metadata was not returned")
                        page = value.get(field)
                        _, rows = self._page(page, expected=None, size=page_size, offset=0)
                        if len({sha256_json(row) for row in rows}) != len(rows):
                            raise LiveProviderError("Connection first page has duplicated rows")
                        self.initial_pages.setdefault(key_for(identifier), page)
                    except LiveProviderError as exc:
                        if failure_category(exc) in {"query_too_expensive", "response_size_limit"}:
                            retry_dense.append(identifier)
                        # Only this connection remains missing. Its existing
                        # page reader performs one bounded individual fallback.
            except LiveProviderError as exc:
                if failure_category(exc) in {"query_too_expensive", "response_size_limit"}:
                    retry_dense = chunk
            if retry_dense and len(chunk) > 1:
                middle = (len(retry_dense) + 1) // 2
                await group(retry_dense[:middle])
                if retry_dense[middle:]: await group(retry_dense[middle:])
            recovery_progress("entities", "running", unit="entities", total_count=len(missing),
                              completed_count=sum(key_for(identifier) in self.initial_pages for identifier in missing),
                              **self.stats())
        batch_size = parameter("retrieval.entity_query_batch_size")
        for start in range(0, len(missing), batch_size):
            await group(missing[start:start + batch_size])

    async def variants(self, variant_ids, kinds):
        scope = tuple(sorted(set(ANALYSIS_VARIANTS.get() or variant_ids)))
        molecular_scope = MOLECULAR_RETRIEVAL_SCOPE.get()
        details = molecular_scope is not None and molecular_scope.phase == "details"
        identifiers, discovery_error = ((list(molecular_scope.study_locus_ids), None) if details
                                        else await self.discover(variant_ids))
        size = parameter("retrieval.credible_set_page_size")
        core_fields = CORE_FIELDS + f" locus(variantIds: $memberVariants, page: {{index: 0, size: {size}}}) {{ count rows {{ {MEMBER_FIELDS} }} }}"
        if EvidenceKind.L2G in kinds:
            core_fields += f" l2GPredictions(page: {{index: 0, size: {size}}}) {{count rows {{ {L2G_FIELDS} }} }}"
        previous_cores = set(self.cores)
        await self._entities(identifiers, root="credibleSet", argument="studyLocusId", fields=core_fields,
                             cache=self.cores, common_variables={"memberVariants": list(scope)})
        for identifier in identifiers:
            core = self.cores.get(identifier, {})
            if identifier not in previous_cores and "locus" in core:
                self.initial_pages.setdefault((identifier, "locus", scope), core["locus"])
            if "l2GPredictions" in core:
                self.initial_pages.setdefault((identifier, "l2GPredictions", ()), core["l2GPredictions"])
        # Cached cores carry members for an earlier variant scope; fetch the new
        # scope's first pages in aliased batches instead of one request each.
        await self._prefetch_member_pages(identifiers, scope)
        if EvidenceKind.L2G in kinds:
            await self._prefetch_l2g_pages(identifiers)
        study_ids = {self.cores[identifier].get("studyId") for identifier in identifiers if identifier in self.cores}
        study_ids.discard(None)
        await self._entities(study_ids, root="study", argument="studyId", fields=STUDY_FIELDS, cache=self.studies)
        membership_budget = [parameter("retrieval.credible_set_max_continuation_requests")]
        coloc_budget = [parameter("retrieval.colocalisation_max_continuation_requests")]
        l2g_budget = [parameter("retrieval.credible_set_max_continuation_requests")]
        by_variant = {variant: [] for variant in variant_ids}
        errors = {variant: discovery_error for variant in variant_ids}
        connections_incomplete = False
        for identifier in identifiers:
            core = self.cores.get(identifier)
            if core is None:
                for variant in errors: errors[variant] = errors[variant] or "credible_set_metadata_unavailable"
                continue
            study = self.studies.get(core.get("studyId"))
            if details and not _detail_scope_allows(core, study, molecular_scope):
                for variant in errors:
                    errors[variant] = errors[variant] or "molecular_detail_scope_unverified"
                continue
            members = await self.connection(identifier, "locus", MEMBER_FIELDS, variant_ids=scope, budget=membership_budget)
            if not members["coverage"]["complete"]:
                for variant in errors: errors[variant] = errors[variant] or "membership_coverage_incomplete"
            detail_members = ([member for member in members["rows"]
                               if member["variant"]["id"] in by_variant
                               and _detail_member_allows(core, study, member["variant"]["id"], molecular_scope)]
                              if details else members["rows"])
            if details and not detail_members:
                # The frozen study may belong to a different variant batch.
                # No source/member tuple means no authorized colocalization query.
                continue
            shared = {**core, "study": study or {}}
            if EvidenceKind.L2G in kinds:
                shared["l2GPredictions"] = await self.connection(identifier, "l2GPredictions", L2G_FIELDS, budget=l2g_budget)
                connections_incomplete |= not shared["l2GPredictions"]["coverage"]["complete"]
            if EvidenceKind.COLOCALIZATION in kinds and (molecular_scope is None or details):
                if native_qtl_study_type(core.get("studyType")) is not None:
                    coloc = deepcopy(await self.connection(identifier, "colocalisation", COLOC_FIELDS, budget=coloc_budget))
                    right_studies = {row.get("otherStudyLocus", {}).get("studyId") for row in coloc["rows"]
                                     if isinstance(row.get("otherStudyLocus"), dict)}
                    right_studies.discard(None)
                    enriched = set()
                    if details:
                        self.study_directory.update({key: self.studies[key] for key in right_studies if key in self.studies})
                        await self._entities(right_studies, root="study", argument="studyId",
                                             fields=STUDY_DIRECTORY_FIELDS, cache=self.study_directory)
                        enriched = {key for key in right_studies
                                    if _trait_relation(self.study_directory.get(key, {}), molecular_scope)
                                    in {"exact_label", "related_label"}}
                        await self._entities(enriched, root="study", argument="studyId", fields=STUDY_FIELDS, cache=self.studies)
                    else:
                        await self._entities(right_studies, root="study", argument="studyId", fields=STUDY_FIELDS, cache=self.studies)
                    for row in coloc["rows"]:
                        other = row.get("otherStudyLocus")
                        if isinstance(other, dict):
                            metadata = self.studies.get(other.get("studyId")) or self.study_directory.get(other.get("studyId"))
                            other["study"] = metadata or {}
                            if details:
                                other["targetTraitRelation"] = _trait_relation(metadata or {}, molecular_scope)
                                other["detailRetrievalState"] = (
                                    "SUCCESS_DATA" if other.get("studyId") in self.studies
                                    else "FAILED" if other.get("studyId") in enriched
                                    else "NOT_REQUESTED" if metadata is not None else "FAILED")
                                if other["detailRetrievalState"] == "FAILED":
                                    coloc["coverage"].update(complete=False, reason="counterpart_study_details_unavailable")
                            if metadata is None:
                                coloc["coverage"].update(complete=False, reason="counterpart_study_metadata_unavailable")
                    shared["colocalisation"] = coloc
                    connections_incomplete |= not coloc["coverage"]["complete"]
                else:
                    shared["colocalisation"] = {"count": 0, "rows": []}
            for member in detail_members:
                variant = member["variant"]["id"]
                if variant not in by_variant:
                    continue
                # One member view retains its own measurements; shared study and
                # connection objects are referenced, not queried for each member.
                by_variant[variant].append({**shared, "locus": {"count": 1, "rows": [member]}})
                if study is None:
                    errors[variant] = errors[variant] or "study_metadata_unavailable"
        variants = [{"id": variant, "credibleSets": {"count": len(by_variant[variant]), "rows": by_variant[variant],
                     "coverage": {"complete": not errors[variant], "reason": errors[variant], "count_scope": "materialized_membership_view"}}}
                    for variant in variant_ids]
        coverage = [{"variant_id": variant, "declared_count": None if errors[variant] else len(by_variant[variant]),
                     "retrieved_count": len(by_variant[variant]), "failure_reason": errors[variant]} for variant in variant_ids]
        recovery_progress("entities", "incomplete" if any(errors.values()) or connections_incomplete else "completed",
                          unit="entities", completed_count=sum(value in self.cores for value in identifiers),
                          total_count=len(identifiers), **self.stats())
        return EntityVariantResult(variants, coverage)


def _detail_scope_allows(core, study, scope):
    """Check actual source metadata before any colocalisation HTTP request."""
    if not isinstance(study, dict) or core.get("studyLocusId") not in scope.study_locus_ids:
        return False
    direct = core.get("qtlGeneId")
    target = study.get("target") or {}
    alternative = target.get("id") if isinstance(target, dict) else None
    if direct and alternative and direct != alternative:
        return False
    gene = direct or alternative
    if gene not in scope.candidate_genes or native_qtl_study_type(core.get("studyType")) is None:
        return False
    biosample = study.get("biosample") or {}
    raw = biosample if isinstance(biosample, dict) else {}
    match = molecular_context_match({"upstream": {"raw": raw}}, scope.context)
    qualified = getattr(scope, "qualified_pairs", ())
    if qualified and not any(pair[:2] == (gene, core.get("studyLocusId")) for pair in qualified):
        return False
    if match["matched"]:
        return True
    return (molecular_context_class(match) == "other_reported_context"
            and any(pair[:2] == (gene, core.get("studyLocusId"))
                    for pair in getattr(scope, "supplementary_pairs", ())))


def _detail_member_allows(core, study, variant, scope):
    """Bind the actual returned member to the frozen gene/study-locus tuple."""
    qualified = getattr(scope, "qualified_pairs", ())
    if not qualified:
        return True  # Compatibility for the existing context-matched detail API.
    target = study.get("target") or {}
    gene = core.get("qtlGeneId") or (target.get("id") if isinstance(target, dict) else None)
    return (gene, core.get("studyLocusId"), variant) in qualified


def _trait_relation(study, scope):
    label = study.get("traitFromSource")
    if not isinstance(label, str) or not label.strip():
        return "unresolved"
    normalized = " ".join(label.casefold().split())
    if any(normalized == " ".join(term.casefold().split()) for term in scope.target_trait_labels):
        return "exact_label"
    if any(context_matches(term, label) for term in scope.target_trait_labels):
        return "related_label"
    return "other_trait"
