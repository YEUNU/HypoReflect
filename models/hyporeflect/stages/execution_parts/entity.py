import re
from typing import Any


class EntitySupport:
    @staticmethod
    def _normalize_metric_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_entity_key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    def _entity_alias_keys(self, entity: Any) -> set[str]:
        raw = str(entity or "").strip().lower()
        key = self._normalize_entity_key(raw)
        if not key:
            return set()
        aliases: set[str] = {key}

        tokens = [tok for tok in re.split(r"[^a-z0-9]+", raw) if tok]
        if not tokens:
            return aliases

        legal_suffixes = {
            "inc",
            "incorporated",
            "corp",
            "corporation",
            "co",
            "company",
            "plc",
            "ltd",
            "llc",
            "lp",
            "sa",
            "ag",
            "nv",
            "group",
            "holdings",
            "holding",
            "the",
        }
        core_tokens = [tok for tok in tokens if tok not in legal_suffixes]
        if core_tokens:
            aliases.add("".join(core_tokens))
            for tok in core_tokens:
                if len(tok) >= 2:
                    aliases.add(tok)
        return aliases

    def _entity_search_aliases(self, entity: Any) -> list[str]:
        raw = str(entity or "").strip()
        if not raw or self._is_generic_entity_label(raw):
            return []
        aliases = []
        seen = set()

        def add(value: Any) -> None:
            text = str(value or "").strip()
            key = text.lower()
            if not key or key in seen:
                return
            seen.add(key)
            aliases.append(text)

        add(raw)
        normalized = self._normalize_entity_key(raw)
        if normalized:
            add(normalized)
        for token in self._entity_alias_keys(raw):
            if len(token) >= 4:
                add(token)
        return aliases

    def _query_entity_candidates(
        self,
        query_state: dict[str, Any],
        user_query: str = "",
    ) -> list[str]:
        candidates: list[str] = []
        top_entity = str(query_state.get("entity", "") or "").strip()
        if top_entity and not self._is_generic_entity_label(top_entity):
            candidates.append(top_entity)
        for slot in self._required_slots(query_state):
            struct = self._parse_slot_struct(slot)
            if not struct:
                continue
            slot_entity = str(struct.get("entity", "") or "").strip()
            if slot_entity and not self._is_generic_entity_label(slot_entity):
                candidates.append(slot_entity)
        deduped: list[str] = []
        seen = set()
        for value in candidates:
            key = str(value or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(value).strip())
        return deduped

    def _entity_matches(self, lhs: Any, rhs: Any) -> bool:
        left = self._entity_alias_keys(lhs)
        right = self._entity_alias_keys(rhs)
        if not left or not right:
            return True
        if not left.isdisjoint(right):
            return True
        for lval in left:
            for rval in right:
                if not lval or not rval:
                    continue
                if lval in rval or rval in lval:
                    return True
        return False

    def _canonical_metric_key(self, metric: Any) -> str:
        text = self._normalize_metric_text(metric)
        if not text:
            return ""
        text = re.sub(r"(?<!\d)(?:fy\s*)?(?:19|20)\d{2}(?!\d)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _metric_matches(self, lhs: Any, rhs: Any) -> bool:
        left = self._canonical_metric_key(lhs)
        right = self._canonical_metric_key(rhs)
        if not left or not right:
            return False
        if left == right:
            return True
        if left in right or right in left:
            return True

        left_terms = {term for term in self._metric_alias_terms(left) if len(term.strip()) >= 4}
        right_terms = {term for term in self._metric_alias_terms(right) if len(term.strip()) >= 4}
        if left_terms and right_terms and not left_terms.isdisjoint(right_terms):
            return True
        return False

    @staticmethod
    def _extract_quarter_tokens(text: str) -> list[str]:
        return re.findall(r"\bq([1-4])\b", str(text or "").lower())

    def _periods_overlap(self, lhs: Any, rhs: Any) -> bool:
        left = str(lhs or "").strip().lower()
        right = str(rhs or "").strip().lower()
        if not left or not right:
            return True
        left_years = set(self._extract_year_tokens(left))
        right_years = set(self._extract_year_tokens(right))
        if left_years and right_years and left_years.isdisjoint(right_years):
            return False
        left_quarters = set(self._extract_quarter_tokens(left))
        right_quarters = set(self._extract_quarter_tokens(right))
        if left_quarters and right_quarters:
            if (
                not left_years
                or not right_years
                or not left_years.isdisjoint(right_years)
            ) and left_quarters.isdisjoint(right_quarters):
                return False
        return True

    @staticmethod
    def _citation_doc_title(citation: str) -> str:
        match = re.search(
            r"^\[\[([^,\]]+),\s*Page\s*\d+\s*,\s*Chunk\s*\d+\s*\]\]$",
            str(citation or "").strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    def _title_year_tokens(self, title: Any) -> list[str]:
        return self._extract_year_tokens(str(title or ""))

    def _filter_nodes_by_query_entity(
        self,
        nodes: list[dict[str, Any]],
        query_state: dict[str, Any],
        *,
        user_query: str = "",
        fail_open: bool = False,
    ) -> list[dict[str, Any]]:
        if not nodes:
            return nodes
        entity_candidates = self._query_entity_candidates(query_state, user_query)
        if not entity_candidates:
            return nodes

        filtered: list[dict[str, Any]] = []
        for node in nodes:
            title = str(node.get("title") or node.get("doc") or "").strip()
            if not title:
                continue
            doc_target = title.split("_", 1)[0].strip()
            if not doc_target:
                continue
            if any(self._entity_matches(candidate, doc_target) for candidate in entity_candidates):
                filtered.append(node)
        if filtered:
            metric_key = self._canonical_metric_key(query_state.get("metric", ""))
            debt_listing_query = any(
                marker in metric_key
                for marker in [
                    "debt securities",
                    "registered to trade",
                    "national securities exchange",
                    "trading symbol",
                ]
            )
            if debt_listing_query:
                return filtered
            period_years = set(self._extract_year_tokens(str(query_state.get("period", "") or "")))
            if period_years:
                period_filtered: list[dict[str, Any]] = []
                for node in filtered:
                    title = str(node.get("title") or node.get("doc") or "").strip()
                    title_years = set(self._title_year_tokens(title))
                    if title_years and title_years.isdisjoint(period_years):
                        continue
                    period_filtered.append(node)
                if period_filtered:
                    return period_filtered
                if not fail_open:
                    return []
            return filtered
        return nodes if fail_open else []

    def _build_entity_retry_entities(
        self,
        query_state: dict[str, Any],
        user_query: str,
    ) -> list[str]:
        entity_candidates = self._query_entity_candidates(query_state, user_query)
        if not entity_candidates:
            return []

        period = str(query_state.get("period", "") or "").strip()
        metric = str(query_state.get("metric", "") or "").strip()
        candidates: list[str] = []
        for entity in entity_candidates:
            entity_aliases = self._entity_search_aliases(entity)
            if not entity_aliases:
                entity_aliases = [entity]
            candidates.extend(entity_aliases)
            for alias in entity_aliases[:2]:
                if period and metric:
                    candidates.append(f"{alias} {period} {metric}")
                elif metric:
                    candidates.append(f"{alias} {metric}")
                if user_query.strip():
                    candidates.append(f"{alias} {user_query.strip()}")

        deduped: list[str] = []
        seen = set()
        for item in candidates:
            key = str(item or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(item).strip())
        return deduped

    async def _entity_guarded_graph_search(
        self,
        *,
        entities: list[str],
        depth: int,
        top_k: int,
        query_state: dict[str, Any],
        user_query: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        txt, data = await self._call_graph_search(entities, depth=depth, top_k=top_k)
        filtered = self._filter_nodes_by_query_entity(data, query_state, user_query=user_query, fail_open=False)
        diagnostics: dict[str, Any] = {
            "initial_raw": len(data),
            "initial_kept": len(filtered),
            "retry_used": False,
            "retry_raw": 0,
            "retry_kept": 0,
            "relaxed_used": False,
            "relaxed_kept": 0,
            "retry_reason": "",
        }
        retry_data: list[dict[str, Any]] = []

        query_entity = str(query_state.get("entity", "") or "").strip()
        answer_type = str(query_state.get("answer_type", "") or "").strip().lower()
        period_years = set(self._extract_year_tokens(str(query_state.get("period", "") or "")))
        kept_years: set[str] = set()
        for node in filtered:
            title = str(node.get("title") or node.get("doc") or "").strip()
            if not title:
                continue
            kept_years.update(self._title_year_tokens(title))
        overlap_year_count = len(period_years & kept_years) if period_years else 0
        compute_sparse_year_coverage = (
            answer_type == "compute"
            and len(period_years) >= 2
            and overlap_year_count < min(2, len(period_years))
        )
        should_retry = diagnostics["initial_kept"] == 0 or compute_sparse_year_coverage
        if (
            query_entity
            and not self._is_generic_entity_label(query_entity)
            and diagnostics["initial_raw"] > 0
            and should_retry
        ):
            retry_entities = self._build_entity_retry_entities(query_state, user_query)
            if retry_entities:
                diagnostics["retry_used"] = True
                diagnostics["retry_reason"] = (
                    "compute_sparse_year_coverage" if compute_sparse_year_coverage else "initial_empty"
                )
                retry_txt, retry_data = await self._call_graph_search(
                    retry_entities,
                    depth=depth,
                    top_k=top_k,
                )
                retry_filtered = self._filter_nodes_by_query_entity(
                    retry_data,
                    query_state,
                    user_query=user_query,
                    fail_open=False,
                )
                diagnostics["retry_raw"] = len(retry_data)
                diagnostics["retry_kept"] = len(retry_filtered)
                if retry_txt:
                    txt = f"{txt}\n\n{retry_txt}" if txt else retry_txt
                if retry_filtered:
                    return txt, retry_filtered, diagnostics

        if not filtered:
            relaxed_source = retry_data if retry_data else data
            if relaxed_source:
                relaxed = self._filter_nodes_by_query_entity(
                    relaxed_source,
                    query_state,
                    user_query=user_query,
                    fail_open=True,
                )
                if relaxed:
                    diagnostics["relaxed_used"] = True
                    diagnostics["relaxed_kept"] = len(relaxed)
                    return txt, relaxed, diagnostics

        return txt, filtered, diagnostics
