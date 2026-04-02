"""
router_tier1.py — Phase 3: Tier 1 Deterministic Regex Router

Responsibilities:
  - Load regex patterns from external config (config/regex_patterns.json)
  - Detect structured entities in a query using pre-compiled regex patterns
  - Update the shared AgentState object with detected entities and routing decision
  - Execute in <5ms deterministic time (no ML inference, no embeddings)

Architecture mandate (Phased Agent Prompt Chain Design.docx):
  - Tier 1 Router = deterministic regex scanning only
  - Execution time: 1–5 milliseconds (bypasses all probabilistic evaluation)
  - Route to "keyword" search IMMEDIATELY when entities are detected
  - Do NOT invoke semantic search from this router
  - Patterns are data-driven (loaded from JSON), not hardcoded
  - Regex patterns compiled ONCE at load time — not inside detect_entities()

Design decisions:
  - load_regex_patterns() returns compiled re.Pattern objects (not raw strings)
    so that detect_entities() pays zero compilation cost at query time
  - tier1_route() returns a NEW state dict (does not mutate the input)
  - If no entities found, route_decision is left UNCHANGED for Tier 2 to decide
  - No imports from keyword_search, semantic_search, or embedder modules

Function signatures (defined before implementation per architecture mandate):

  load_regex_patterns(config_path: str)
      -> dict[str, re.Pattern]
      Loads and compiles all regex patterns from config JSON.

  detect_entities(query: str, patterns: dict[str, re.Pattern])
      -> list[str]
      Applies all compiled patterns to query; returns deduplicated UPPERCASE matches.

  tier1_route(state: dict, patterns: dict[str, re.Pattern])
      -> dict
      Updates state with detected_entities and conditionally sets route_decision.

Public API:
  load_regex_patterns(config_path)          -> dict[str, re.Pattern]
  detect_entities(query, patterns)          -> list[str]
  tier1_route(state, patterns)              -> dict
"""

import re
import json
import copy
import time


# ──────────────────────────────────────────────────────────────
# 1. Pattern loader
# ──────────────────────────────────────────────────────────────

def load_regex_patterns(config_path: str) -> dict[str, re.Pattern]:
    """
    Load regex patterns from a JSON config file and compile them.

    Why compile here and not in detect_entities()?
        re.compile() has a small but non-trivial cost on each call.
        Since detect_entities() is called on every query, compiling during
        load (once at startup) ensures <5ms detection latency at query time.

    Config format — supports both simple and extended schema:
        Simple  : { "course_code": "pattern_string" }
        Extended: { "course_code": { "pattern": "...", "description": "..." } }

    Args:
        config_path: Absolute or relative path to the JSON patterns file.

    Returns:
        dict[str, re.Pattern]: Maps pattern_name → compiled regex object.
        Keys prefixed with "_" (comments) are ignored.

    Raises:
        FileNotFoundError: If config_path does not exist.
        re.error: If any pattern string is an invalid regex.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    compiled: dict[str, re.Pattern] = {}

    for name, value in raw_config.items():
        # Skip comment keys (convention: keys starting with "_")
        if name.startswith("_"):
            continue

        # Support both simple string and extended dict schema
        if isinstance(value, str):
            pattern_str = value
        elif isinstance(value, dict):
            if "pattern" not in value:
                raise ValueError(
                    f"Pattern '{name}' in config is missing required 'pattern' key."
                )
            pattern_str = value["pattern"]
        else:
            raise ValueError(
                f"Pattern '{name}' has unsupported value type: {type(value).__name__}. "
                f"Expected str or dict."
            )

        # Compile with IGNORECASE so query input matching is case-insensitive.
        # The matched text is returned in UPPERCASE regardless (handled in detect_entities).
        compiled[name] = re.compile(pattern_str, re.IGNORECASE)

    return compiled


# ──────────────────────────────────────────────────────────────
# 2. Entity detector
# ──────────────────────────────────────────────────────────────

def detect_entities(
    query: str,
    patterns: dict[str, re.Pattern],
) -> list[str]:
    """
    Apply all compiled regex patterns to the query and extract matched entities.

    Each pattern is applied independently to the full query string.
    All matches from all patterns are collected, deduplicated, and returned
    in UPPERCASE to ensure stable downstream key comparison.

    Why UPPERCASE?
        Canonical form for identifiers (CS-202 not cs-202) ensures consistent
        display in rationale strings and detected_entities list, regardless of
        how the user typed the query ("cs-202", "Cs-202", "CS-202").

    Architecture constraint:
        No ML inference. No string hardcoding. Pure regex.
        This function must never reference specific codes like "CS-202".

    Args:
        query:    Raw query string from the user.
        patterns: Pre-compiled regex patterns from load_regex_patterns().

    Returns:
        list[str]: Deduplicated list of matched entity strings in UPPERCASE.
                   Ordered by position of first occurrence in the query.
                   Empty list if no entities detected.

    Examples:
        detect_entities("CS-202 prerequisites", compiled_patterns)
            -> ["CS-202"]

        detect_entities("CS-202 and ENG-404 eligibility", compiled_patterns)
            -> ["CS-202", "ENG-404"]

        detect_entities("What happens if I miss an exam?", compiled_patterns)
            -> []
    """
    seen: set[str] = set()
    # Collect (start_pos, entity_str) to preserve left-to-right order
    ordered: list[tuple[int, str]] = []

    for pattern_name, compiled_pattern in patterns.items():
        for match in compiled_pattern.finditer(query):
            entity = match.group().upper()
            if entity not in seen:
                seen.add(entity)
                ordered.append((match.start(), entity))

    # Sort by position of first occurrence (left-to-right in query)
    ordered.sort(key=lambda x: x[0])
    return [entity for _, entity in ordered]


# ──────────────────────────────────────────────────────────────
# 3. Tier 1 routing function
# ──────────────────────────────────────────────────────────────

def tier1_route(
    state: dict,
    patterns: dict[str, re.Pattern],
) -> dict:
    """
    Apply Tier 1 deterministic routing to the shared AgentState.

    Routing logic:
        IF entities detected  → set route_decision = "keyword"
                                 set detected_entities = [matched list]
                                 set rationale = explanation string
        IF no entities found  → return state unchanged
                                 (Tier 2 router will handle this query)

    This function DOES NOT:
        - Call keyword_search or semantic_search
        - Use any ML model or embedding
        - Modify any field other than detected_entities, route_decision, rationale

    Performance contract:
        regex.finditer on 30 patterns against a short query string executes
        in microseconds — well within the <5ms architecture mandate.

    Args:
        state:    Shared state dict conforming to the data contract.
                  Must contain at minimum: "query" key.
        patterns: Pre-compiled patterns from load_regex_patterns().

    Returns:
        dict: Updated state copy (input state is NOT mutated).
              If entities found:
                  detected_entities = ["CS-202", ...]
                  route_decision    = "keyword"
                  rationale         = "Exact match search activated due to
                                       detection of code(s): ['CS-202']."
              If no entities found:
                  All fields returned as-is (unchanged from input).
    """
    # ── Time the detection for performance logging ─────────────
    t_start = time.perf_counter()

    query = state.get("query", "")
    entities = detect_entities(query, patterns)

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── Always return a new state copy, never mutate input ─────
    new_state = copy.deepcopy(state)
    new_state["_tier1_latency_ms"] = round(t_elapsed_ms, 4)   # internal diagnostic key

    if not entities:
        # No structured entities detected — leave route_decision for Tier 2
        # Rationale is not updated here; Tier 2 will write its own
        return new_state

    # ── Entities found — route deterministically to keyword ────
    new_state["detected_entities"] = entities
    new_state["route_decision"]    = "keyword"
    new_state["rationale"]         = (
        f"Tier 1 (Regex): Exact match search activated due to detection of "
        f"code(s): {entities}. "
        f"Bypassing semantic evaluation. "
        f"Detection latency: {t_elapsed_ms:.3f}ms."
    )

    return new_state
