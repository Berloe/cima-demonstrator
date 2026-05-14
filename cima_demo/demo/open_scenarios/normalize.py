from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .catalog import DATASET_SPECS, get_dataset_spec, resolve_dataset_ids
from .models import NormalizationResult, OpenCaseDocument, OpenCaseEvidence, OpenScenarioCase


def _require_hf_datasets() -> Any:
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The optional dependency 'datasets' is required for Hugging Face-backed open scenarios. "
            "Install it with: pip install -r requirements-open-scenarios.txt"
        ) from exc
    return load_from_disk




def _dataset_splits(dataset: Any) -> list[str]:
    try:
        return list(dataset.keys())
    except Exception:
        return []


def _resolve_split(dataset: Any, requested_split: str, *, dataset_id: str) -> tuple[Any, str]:
    splits = _dataset_splits(dataset)
    if requested_split in splits:
        return dataset[requested_split], requested_split
    aliases = {
        "validation": ["validation", "val", "dev", "test", "train"],
        "test": ["test", "validation", "val", "dev", "train"],
        "train": ["train", "validation", "val", "dev", "test"],
    }
    for candidate in aliases.get(requested_split, [requested_split, "validation", "val", "dev", "test", "train"]):
        if candidate in splits:
            return dataset[candidate], candidate
    raise KeyError(
        f"Dataset '{dataset_id}' does not contain requested split '{requested_split}' and no fallback split is available. "
        f"Available splits: {splits}"
    )

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_json_files(root: Path) -> Iterable[Path]:
    for ext in ("*.json", "*.jsonl"):
        yield from sorted(root.glob(ext))


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"value": item} for item in value]
    if isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values()):
        length = max(len(v) for v in value.values())
        rows: list[dict[str, Any]] = []
        for idx in range(length):
            rows.append({key: (values[idx] if idx < len(values) else None) for key, values in value.items()})
        return rows
    if isinstance(value, dict):
        return [value]
    return [{"value": value}]



def _non_empty_string(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen.keys())


def _split_candidates(requested_split: str) -> list[str]:
    aliases = {
        "validation": ["validation", "val", "dev", "test", "train"],
        "test": ["test", "validation", "val", "dev", "train"],
        "train": ["train", "validation", "val", "dev", "test"],
    }
    candidates = aliases.get(requested_split, [requested_split, "validation", "val", "dev", "test", "train"])
    return list(dict.fromkeys(candidates))

def _stringify_turn(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if isinstance(turn, dict):
        speaker = str(
            turn.get("speaker")
            or turn.get("speaker_name")
            or turn.get("role")
            or turn.get("participant")
            or "Speaker"
        ).strip()
        text = str(
            turn.get("content")
            or turn.get("text")
            or turn.get("utterance")
            or turn.get("sentence")
            or ""
        ).strip()
        if not text:
            return speaker
        return f"{speaker}: {text}"
    return str(turn).strip()


def _render_meeting_transcript(record: dict[str, Any]) -> tuple[str, list[str]]:
    raw_turns = record.get("meeting_transcripts") or record.get("transcript") or record.get("dialogue") or []
    turns = [_stringify_turn(turn) for turn in raw_turns if _stringify_turn(turn)]
    return "\n".join(turns), turns


def _slice_turn_ranges(turns: list[str], ranges: Any) -> list[str]:
    snippets: list[str] = []
    for pair in ranges or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            start = int(pair[0])
            end = int(pair[1])
        except Exception:
            continue
        if start < 0:
            continue
        end = min(end, len(turns) - 1)
        if start > end:
            continue
        snippets.append("\n".join(turns[start : end + 1]))
    return [snippet for snippet in snippets if snippet.strip()]


def _qmsum_cases(raw_root: Path, *, split: str, limit: int | None, seed: int) -> list[OpenScenarioCase]:
    repo_root = raw_root / "qmsum" / "repo" / "data" / "ALL"
    split_root = repo_root / split
    if not split_root.exists():
        if split == "validation" and (repo_root / "val").exists():
            split_root = repo_root / "val"
        elif split == "test" and (repo_root / "test").exists():
            split_root = repo_root / "test"
        elif split == "train" and (repo_root / "train").exists():
            split_root = repo_root / "train"
        elif (repo_root / "all").exists():
            split_root = repo_root / "all"
    files = list(_iter_json_files(split_root))
    rng = random.Random(seed)
    if limit is not None and len(files) > limit:
        files = rng.sample(files, limit)
    cases: list[OpenScenarioCase] = []
    for path in files:
        record = _read_json(path)
        transcript_text, turns = _render_meeting_transcript(record)
        title = str(path.stem)
        document = OpenCaseDocument(doc_id=f"{title}-transcript", title=title, text=transcript_text)
        query_groups = [
            ("general", record.get("general_query_list") or []),
            ("specific", record.get("specific_query_list") or []),
        ]
        for group, entries in query_groups:
            for idx, entry in enumerate(entries):
                query = str(entry.get("query") or "").strip()
                answer = str(entry.get("answer") or "").strip()
                if not query or not answer:
                    continue
                evidence = [
                    OpenCaseEvidence(
                        doc_id=document.doc_id,
                        text=snippet,
                        metadata={"evidence_kind": "relevant_text_span", "query_group": group},
                    )
                    for snippet in _slice_turn_ranges(turns, entry.get("relevant_text_span"))
                ]
                cases.append(
                    OpenScenarioCase(
                        case_id=f"qmsum::{title}::{group}::{idx}",
                        dataset_id="qmsum",
                        task_family="meeting_query_summary",
                        split=split,
                        prompt=query,
                        gold_answers=[answer],
                        documents=[document],
                        gold_evidence=evidence,
                        metadata={"query_group": group, "source_path": str(path)},
                    )
                )
    return cases


def _find_explainmeetsum_root(raw_root: Path) -> Path | None:
    candidates = [
        raw_root / "explainmeetsum" / "repo" / "data" / "ExplainMeetSum",
        raw_root / "explainmeetsum" / "repo" / "ExplainMeetSum",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_evidence_strings(value: Any) -> list[str]:
    """Extract evidence text from nested dataset records.

    ExplainMeetSum nests evidence under explainable_answer[].evidence[].content,
    while QASPER uses answer.evidence / answer.highlighted_evidence. The helper
    follows only evidence branches and emits textual evidence fields; it avoids
    leaking labels such as CES or answer_sentence into evidence snippets.
    """
    strings: list[str] = []
    text_keys = {"content", "text", "sentence", "snippet", "dialogue_sentence", "highlighted_evidence"}

    def append_text(node: Any) -> None:
        text = _non_empty_string(node)
        if text:
            strings.append(text)

    def walk(node: Any, *, evidence_context: bool = False) -> None:
        if isinstance(node, str):
            if evidence_context:
                append_text(node)
            return
        if isinstance(node, list):
            for item in node:
                walk(item, evidence_context=evidence_context)
            return
        if not isinstance(node, dict):
            return
        for key, child in node.items():
            key_lower = str(key).lower()
            if "evidence" in key_lower:
                walk(child, evidence_context=True)
                continue
            if evidence_context and key_lower in text_keys:
                if isinstance(child, str):
                    append_text(child)
                else:
                    walk(child, evidence_context=True)
                continue
            if not evidence_context:
                walk(child, evidence_context=False)

    walk(value)
    return _dedupe_strings(strings)


def _explainmeetsum_query_entries(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return query entries from native ExplainMeetSum and QMSum-like records."""
    containers: list[tuple[str, dict[str, Any]]] = [("root", record)]
    for key in ("explainable_qmsum", "explainable_ami", "explainable_icsi"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append((key, value))
    entries: list[tuple[str, dict[str, Any]]] = []
    for container_name, container in containers:
        for group in ("general_query_list", "specific_query_list"):
            raw_entries = container.get(group) or []
            for entry in raw_entries:
                if isinstance(entry, dict):
                    entries.append((f"{container_name}:{group}", entry))
    return entries


def _canonicalize_explainmeetsum_prompt(query: str, *, answer: str, evidence_texts: list[str], transcript_text: str) -> tuple[str, dict[str, Any]]:
    """Fix known benchmark typos without hiding that normalization changed text.

    ExplainMeetSum/QMSum material contains an education-domain typo where
    "intuition in a private setting" denotes private/home tuition. Leaving the
    typo in the prompt steers instruction models toward psychological
    "intuition" instead of educational provision. We only canonicalize this
    when the surrounding answer/evidence/transcript makes the tuition meaning
    explicit or the query itself contains the private-setting pattern.
    """
    original = (query or "").strip()
    lowered = original.lower()
    if "intuition" not in lowered:
        return original, {}
    support_text = "\n".join([answer or "", *evidence_texts, transcript_text[:6000]]).lower()
    combined = f"{lowered}\n{support_text}"
    tuition_context = any(
        re.search(pattern, combined)
        for pattern in (
            r"\btuition\b",
            r"\bhome\s+tuition\b",
            r"\bprivate\s+tuition\b",
            r"\bprivate\s+setting\b",
            r"\bindividual\s+provision\b",
            r"\balternative\s+provision\b",
        )
    )
    if not tuition_context:
        return original, {}
    canonical = original.replace("intuition", "tuition").replace("Intuition", "Tuition")
    if canonical == original:
        return original, {}
    return canonical, {
        "prompt_canonicalized": True,
        "original_prompt": original,
        "canonicalization_rule": "explainmeetsum_education_intuition_to_tuition",
    }


def _explainmeetsum_cases_for_split(
    explain_root: Path,
    *,
    requested_split: str,
    source_split: str,
    limit: int | None,
    seed: int,
) -> list[OpenScenarioCase]:
    split_root = explain_root / source_split
    files = list(_iter_json_files(split_root))
    rng = random.Random(seed)
    if limit is not None and len(files) > limit:
        files = rng.sample(files, limit)
    cases: list[OpenScenarioCase] = []
    for path in files:
        record = _read_json(path)
        transcript_text, _turns = _render_meeting_transcript(record)
        if not transcript_text.strip():
            continue
        title = str(path.stem)
        document = OpenCaseDocument(doc_id=f"{title}-transcript", title=title, text=transcript_text)
        for idx, (group, entry) in enumerate(_explainmeetsum_query_entries(record)):
            query = str(entry.get("query") or "").strip()
            answer = str(entry.get("answer") or "").strip()
            if not query or not answer:
                continue
            evidence_texts = _extract_evidence_strings(entry)
            query, canonicalization_metadata = _canonicalize_explainmeetsum_prompt(
                query,
                answer=answer,
                evidence_texts=evidence_texts,
                transcript_text=transcript_text,
            )
            evidence = [
                OpenCaseEvidence(doc_id=document.doc_id, text=text, metadata={"evidence_kind": "summary_evidence", "query_group": group})
                for text in evidence_texts
            ]
            cases.append(
                OpenScenarioCase(
                    case_id=f"explainmeetsum::{title}::{source_split}::{idx}",
                    dataset_id="explainmeetsum",
                    task_family="explainable_meeting_summary",
                    split=requested_split,
                    prompt=query,
                    gold_answers=[answer],
                    documents=[document],
                    gold_evidence=evidence,
                    metadata={"source_path": str(path), "source_split": source_split, "query_group": group, **canonicalization_metadata},
                )
            )
    return cases


def _explainmeetsum_cases(raw_root: Path, *, split: str, limit: int | None, seed: int) -> list[OpenScenarioCase]:
    explain_root = _find_explainmeetsum_root(raw_root)
    if explain_root is None:
        raise FileNotFoundError(
            "ExplainMeetSum raw repository was downloaded, but built dataset files were not found. "
            "Run the repository conversion step or place the built dataset under data/ExplainMeetSum."
        )
    available = [path.name for path in explain_root.iterdir() if path.is_dir()]
    split_candidates = [candidate for candidate in _split_candidates(split) if candidate in available]
    if split == "validation" and "val" in available and "val" not in split_candidates:
        split_candidates.insert(0, "val")
    if not split_candidates:
        raise FileNotFoundError(
            f"ExplainMeetSum split '{split}' is not available under {explain_root}. Available splits: {available}"
        )
    attempted: dict[str, int] = {}
    for idx, source_split in enumerate(split_candidates):
        cases = _explainmeetsum_cases_for_split(
            explain_root,
            requested_split=split,
            source_split=source_split,
            limit=limit,
            seed=seed + idx,
        )
        attempted[source_split] = len(cases)
        if cases:
            return cases
    raise RuntimeError(f"ExplainMeetSum normalization produced zero cases. Attempted splits: {attempted}")



def _take_columnar_value(value: Any, idx: int) -> Any:
    """Take item idx from HF Datasets columnar nested structures.

    The QASPER parquet conversion may expose `qas` as a dict of columns rather
    than as a list of question dicts. Nested answer annotations can be columnar
    as well. This helper reconstructs the row-shaped object for one question
    without assuming a single serialization shape.
    """
    if isinstance(value, list):
        return value[idx] if 0 <= idx < len(value) else None
    if isinstance(value, dict):
        return {key: _take_columnar_value(child, idx) for key, child in value.items()}
    return value


def _qasper_question_count(qas: dict[str, Any]) -> int:
    for key in ("question", "question_id", "search_query", "question_writer"):
        value = qas.get(key)
        if isinstance(value, list):
            return len(value)
    answers = qas.get("answers")
    if isinstance(answers, list):
        return len(answers)
    return 0


def _qasper_qas_rows(qas: Any) -> list[dict[str, Any]]:
    if isinstance(qas, list):
        return [item if isinstance(item, dict) else {"value": item} for item in qas]
    if not isinstance(qas, dict):
        return []
    count = _qasper_question_count(qas)
    if count <= 0:
        return _as_rows(qas)
    return [{key: _take_columnar_value(value, idx) for key, value in qas.items()} for idx in range(count)]


def _is_answer_axis_list(key: str, value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if key in {"unanswerable", "yes_no", "free_form_answer"}:
        return True
    if key in {"extractive_spans", "evidence", "highlighted_evidence"}:
        # A single answer uses list[str]. A columnar list of answers uses list[list[str]].
        return any(isinstance(item, (list, dict)) for item in value)
    return False


def _expand_qasper_answer_payload(payload: Any) -> list[Any]:
    """Return answer-annotation-shaped records from QASPER answer payloads.

    Handles all shapes observed across the legacy HF script and parquet viewer:
    list[annotation], {answer: list[answer]}, {answer: dict-of-lists}, and a
    single answer dict.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        expanded: list[Any] = []
        for item in payload:
            expanded.extend(_expand_qasper_answer_payload(item))
        return expanded
    if not isinstance(payload, dict):
        return [payload]

    if "answer" in payload:
        answer_value = payload.get("answer")
        if isinstance(answer_value, list):
            rows: list[dict[str, Any]] = []
            for idx, answer_item in enumerate(answer_value):
                row: dict[str, Any] = {"answer": answer_item}
                for meta_key in ("annotation_id", "worker_id"):
                    meta_value = payload.get(meta_key)
                    if isinstance(meta_value, list):
                        row[meta_key] = meta_value[idx] if idx < len(meta_value) else None
                    elif meta_value is not None:
                        row[meta_key] = meta_value
                rows.append(row)
            return rows
        if isinstance(answer_value, dict):
            axis_keys = [key for key, value in answer_value.items() if _is_answer_axis_list(key, value)]
            # If answer fields are columnar over annotations, expand them.
            if axis_keys:
                n = max(len(answer_value[key]) for key in axis_keys if isinstance(answer_value.get(key), list))
                rows = []
                for idx in range(n):
                    answer_row: dict[str, Any] = {}
                    for key, value in answer_value.items():
                        if _is_answer_axis_list(key, value):
                            answer_row[key] = value[idx] if idx < len(value) else None
                        else:
                            answer_row[key] = value
                    row = {"answer": answer_row}
                    for meta_key in ("annotation_id", "worker_id"):
                        meta_value = payload.get(meta_key)
                        if isinstance(meta_value, list):
                            row[meta_key] = meta_value[idx] if idx < len(meta_value) else None
                        elif meta_value is not None:
                            row[meta_key] = meta_value
                    rows.append(row)
                return rows
        return [payload]

    # Some parquet shapes pass the answer dict directly, with fields possibly
    # columnar across annotations.
    axis_keys = [key for key, value in payload.items() if _is_answer_axis_list(key, value)]
    if axis_keys:
        n = max(len(payload[key]) for key in axis_keys if isinstance(payload.get(key), list))
        rows = []
        for idx in range(n):
            row = {}
            for key, value in payload.items():
                row[key] = value[idx] if _is_answer_axis_list(key, value) and idx < len(value) else value
            rows.append(row)
        return rows
    return [payload]

def _qasper_document_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    sections = []
    for item in _as_rows(row.get("full_text")):
        section_name = str(item.get("section_name") or item.get("section") or "").strip()
        paragraphs = item.get("paragraphs") or item.get("paragraph") or item.get("text") or []
        if isinstance(paragraphs, str):
            para_text = paragraphs.strip()
        else:
            para_text = "\n".join(str(p).strip() for p in paragraphs if str(p).strip())
        if not para_text:
            continue
        sections.append(f"## {section_name}\n{para_text}" if section_name else para_text)
    pieces = [piece for piece in [title, abstract, "\n\n".join(sections)] if piece]
    return "\n\n".join(pieces)


def _qasper_single_answer_text(answer: dict[str, Any]) -> str | None:
    nested = answer.get("answer") if isinstance(answer.get("answer"), dict) else answer
    if not isinstance(nested, dict):
        return _non_empty_string(nested)
    if nested.get("unanswerable") is True:
        return "Unanswerable"
    free_form_value = nested.get("free_form_answer")
    free_form = free_form_value.strip() if isinstance(free_form_value, str) else None
    if free_form:
        return free_form
    extractive = nested.get("extractive_spans")
    if isinstance(extractive, list):
        spans = [text for item in extractive if (text := _non_empty_string(item))]
        if spans:
            return "; ".join(spans)
    yes_no = nested.get("yes_no")
    if isinstance(yes_no, bool):
        return "Yes" if yes_no else "No"
    for key in ("text", "answer"):
        text = nested.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None

def _answer_texts(value: Any) -> tuple[list[str], list[str]]:
    answers: list[str] = []
    evidence: list[str] = []

    def collect_answer(node: Any) -> None:
        for answer_record in _expand_qasper_answer_payload(node):
            if isinstance(answer_record, list):
                collect_answer(answer_record)
                continue
            if isinstance(answer_record, dict):
                text = _qasper_single_answer_text(answer_record)
                if text:
                    answers.append(text)
                evidence_sources = [answer_record]
                if isinstance(answer_record.get("answer"), dict):
                    evidence_sources.append(answer_record["answer"])
                for evidence_source in evidence_sources:
                    if not isinstance(evidence_source, dict):
                        continue
                    evidence.extend(_extract_evidence_strings({"evidence": evidence_source.get("evidence") or []}))
                    evidence.extend(
                        _extract_evidence_strings({"highlighted_evidence": evidence_source.get("highlighted_evidence") or []})
                    )
                if isinstance(answer_record.get("all_answers"), list):
                    collect_answer(answer_record["all_answers"])
                continue
            text = _non_empty_string(answer_record)
            if text:
                answers.append(text)

    collect_answer(value)
    return _dedupe_strings(answers), _dedupe_strings(evidence)

def _qasper_cases_from_dataset(
    dataset: Any,
    *,
    requested_split: str,
    source_split: str,
    limit: int | None,
    seed: int,
) -> list[OpenScenarioCase]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if limit is not None and len(indices) > limit:
        indices = rng.sample(indices, limit)
    cases: list[OpenScenarioCase] = []
    for idx in indices:
        row = dict(dataset[int(idx)])
        doc_id = str(row.get("id") or f"qasper-doc-{idx}")
        document = OpenCaseDocument(doc_id=doc_id, title=str(row.get("title") or doc_id), text=_qasper_document_text(row))
        if not document.text.strip():
            continue
        for q_idx, qa in enumerate(_qasper_qas_rows(row.get("qas"))):
            prompt = str(qa.get("question") or qa.get("query") or "").strip()
            if not prompt:
                continue
            answers, evidence_texts = _answer_texts(qa.get("answers") or qa.get("answer") or qa)
            if not answers:
                continue
            qid = qa.get("question_id") or q_idx
            cases.append(
                OpenScenarioCase(
                    case_id=f"qasper::{doc_id}::{qid}",
                    dataset_id="qasper",
                    task_family="evidence_qa",
                    split=requested_split,
                    prompt=prompt,
                    gold_answers=answers,
                    documents=[document],
                    gold_evidence=[
                        OpenCaseEvidence(doc_id=doc_id, text=text, metadata={"evidence_kind": "answer_support"})
                        for text in evidence_texts
                    ],
                    metadata={"paper_id": doc_id, "source_split": source_split},
                )
            )
    return cases


def _qasper_cases(raw_root: Path, *, split: str, limit: int | None, seed: int) -> list[OpenScenarioCase]:
    load_from_disk = _require_hf_datasets()
    ds_root = raw_root / "qasper" / "hf"
    dataset_dict = load_from_disk(str(ds_root))
    available_splits = _dataset_splits(dataset_dict)
    split_name = split if split != "validation" else "validation"
    candidates = [candidate for candidate in _split_candidates(split_name) if candidate in available_splits]
    if not candidates:
        raise KeyError(f"QASPER has no usable split for '{split}'. Available splits: {available_splits}")
    attempted: dict[str, int] = {}
    for idx, source_split in enumerate(candidates):
        dataset = dataset_dict[source_split]
        cases = _qasper_cases_from_dataset(
            dataset,
            requested_split=split,
            source_split=source_split,
            limit=limit,
            seed=seed + idx,
        )
        attempted[source_split] = len(cases)
        if cases:
            return cases
    raise RuntimeError(f"QASPER normalization produced zero answerable cases. Attempted splits: {attempted}")


def _hotpot_context_documents(row: dict[str, Any]) -> tuple[list[OpenCaseDocument], list[OpenCaseEvidence]]:
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    documents: list[OpenCaseDocument] = []
    doc_map: dict[str, str] = {}
    for idx, title in enumerate(titles):
        sents = sentences[idx] if idx < len(sentences) else []
        text = " ".join(str(sentence).strip() for sentence in sents if str(sentence).strip())
        doc_id = f"ctx-{idx}"
        documents.append(OpenCaseDocument(doc_id=doc_id, title=str(title), text=text))
        doc_map[str(title)] = doc_id
    evidence: list[OpenCaseEvidence] = []
    supporting = row.get("supporting_facts") or {}
    supp_titles = supporting.get("title") or []
    supp_sent_ids = supporting.get("sent_id") or []
    for idx, title in enumerate(supp_titles):
        sent_id = supp_sent_ids[idx] if idx < len(supp_sent_ids) else None
        title_str = str(title)
        if title_str not in doc_map:
            continue
        doc_idx = titles.index(title) if title in titles else None
        snippet = ""
        if doc_idx is not None and sent_id is not None and doc_idx < len(sentences):
            doc_sents = sentences[doc_idx]
            if isinstance(sent_id, int) and 0 <= sent_id < len(doc_sents):
                snippet = str(doc_sents[sent_id]).strip()
        if snippet:
            evidence.append(
                OpenCaseEvidence(doc_id=doc_map[title_str], text=snippet, metadata={"evidence_kind": "supporting_fact", "title": title_str})
            )
    return documents, evidence


def _hotpot_cases(raw_root: Path, *, split: str, limit: int | None, seed: int) -> list[OpenScenarioCase]:
    load_from_disk = _require_hf_datasets()
    ds_root = raw_root / "hotpotqa" / "hf"
    split_name = "validation" if split == "validation" else split
    dataset_dict = load_from_disk(str(ds_root))
    dataset, source_split = _resolve_split(dataset_dict, split_name, dataset_id="hotpotqa")
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if limit is not None and len(indices) > limit:
        indices = rng.sample(indices, limit)
    cases: list[OpenScenarioCase] = []
    for idx in indices:
        row = dict(dataset[int(idx)])
        documents, evidence = _hotpot_context_documents(row)
        if not documents:
            continue
        answer = str(row.get("answer") or "").strip()
        prompt = str(row.get("question") or "").strip()
        if not prompt or not answer:
            continue
        cases.append(
            OpenScenarioCase(
                case_id=f"hotpotqa::{row.get('id', idx)}",
                dataset_id="hotpotqa",
                task_family="multi_hop_qa",
                split=split,
                prompt=prompt,
                gold_answers=[answer],
                documents=documents,
                gold_evidence=evidence,
                metadata={"level": row.get("level"), "question_type": row.get("type"), "source_split": source_split},
            )
        )
    return cases


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def _fever_split_candidates(requested_split: str) -> list[str]:
    """Return direct FEVER file stems to try for a requested split.

    FEVER's official shared-task test/dev files are partly unlabelled. For an
    executable scenario we need a gold label, so requested `test` first tries
    paper_test (labelled in the official release), then falls back to labelled
    development/train material. Unlabelled files are only considered last and
    their rows are filtered out because they lack `label`.
    """
    mapping = {
        "test": ["paper_test", "labelled_dev", "paper_dev", "train", "unlabelled_test"],
        "validation": ["labelled_dev", "paper_dev", "train", "paper_test"],
        "val": ["labelled_dev", "paper_dev", "train", "paper_test"],
        "dev": ["labelled_dev", "paper_dev", "train", "paper_test"],
        "train": ["train", "labelled_dev", "paper_dev", "paper_test"],
    }
    return list(dict.fromkeys(mapping.get(requested_split, [requested_split, "labelled_dev", "train", "paper_test"])))


def _normalize_fever_evidence_refs(value: Any) -> list[dict[str, Any]]:
    """Flatten FEVER evidence annotations without pretending to resolve text."""
    refs: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            # Common FEVER atom: [annotation_id, evidence_id, page, sentence_id]
            if len(node) >= 4 and not any(isinstance(item, (list, dict)) for item in node[:4]):
                refs.append(
                    {
                        "annotation_id": node[0],
                        "evidence_id": node[1],
                        "page": node[2],
                        "sentence_id": node[3],
                    }
                )
                return
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            refs.append(dict(node))

    walk(value)
    # Deterministic de-duplication while preserving order.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for ref in refs:
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(ref)
    return deduped


def _fever_cases_for_file(
    path: Path,
    *,
    requested_split: str,
    source_split: str,
    limit: int | None,
    seed: int,
) -> list[OpenScenarioCase]:
    rows = list(_iter_jsonl(path))
    labelled = [row for row in rows if _non_empty_string(row.get("claim")) and _non_empty_string(row.get("label"))]
    rng = random.Random(seed)
    if limit is not None and len(labelled) > limit:
        labelled = rng.sample(labelled, limit)
    cases: list[OpenScenarioCase] = []
    for idx, row in enumerate(labelled):
        claim = str(row.get("claim") or "").strip()
        label = str(row.get("label") or "").strip().upper()
        if not claim or not label:
            continue
        row_id = str(row.get("id") or f"{source_split}-{idx}")
        evidence_refs = _normalize_fever_evidence_refs(row.get("evidence"))
        evidence_ref_lines = [
            f"- page={ref.get('page')} sentence_id={ref.get('sentence_id')} evidence_id={ref.get('evidence_id')}"
            for ref in evidence_refs[:8]
            if ref.get("page") is not None or ref.get("sentence_id") is not None
        ]
        evidence_notice = (
            "Resolved Wikipedia evidence text is not bundled in this demonstrator. "
            "The following FEVER evidence references are metadata only."
        )
        document_text = "\n".join(
            part
            for part in [
                f"Claim: {claim}",
                evidence_notice,
                "Evidence references:\n" + "\n".join(evidence_ref_lines) if evidence_ref_lines else "Evidence references: none materialized.",
            ]
            if part
        )
        prompt = (
            "Classify the FEVER claim using only the provided local material. "
            "Return one label from SUPPORTS, REFUTES, or NOT ENOUGH INFO, and state when resolved evidence text is unavailable.\n"
            f"Claim: {claim}"
        )
        document = OpenCaseDocument(
            doc_id=f"fever-{row_id}-claim",
            title=f"FEVER claim {row_id}",
            text=document_text,
            metadata={"evidence_mode": "unresolved_references", "source_split": source_split},
        )
        cases.append(
            OpenScenarioCase(
                case_id=f"fever::{source_split}::{row_id}",
                dataset_id="fever",
                task_family="claim_verification_claim_only",
                split=requested_split,
                prompt=prompt,
                gold_answers=[label],
                gold_label=label,
                documents=[document],
                gold_evidence=[],
                metadata={
                    "source_split": source_split,
                    "source_path": str(path),
                    "original_id": row_id,
                    "verifiable": row.get("verifiable"),
                    "evidence_mode": "unresolved_references",
                    "evidence_refs": evidence_refs[:20],
                },
            )
        )
    return cases


def _fever_cases(raw_root: Path, *, split: str, limit: int | None, seed: int) -> list[OpenScenarioCase]:
    fever_root = raw_root / "fever" / "raw"
    if not fever_root.exists():
        raise FileNotFoundError(
            f"FEVER direct-files download was not found at {fever_root}. Run cima_demo_open_download first."
        )
    attempted: dict[str, int] = {}
    for idx, source_split in enumerate(_fever_split_candidates(split)):
        path = fever_root / f"{source_split}.jsonl"
        if not path.exists():
            continue
        cases = _fever_cases_for_file(
            path,
            requested_split=split,
            source_split=source_split,
            limit=limit,
            seed=seed + idx,
        )
        attempted[source_split] = len(cases)
        if cases:
            return cases
    raise RuntimeError(
        "FEVER normalization produced zero labelled claim-only cases. "
        f"Attempted files: {attempted}. Unlabelled FEVER files are not executable."
    )


BUILDERS = {
    "qmsum": _qmsum_cases,
    "explainmeetsum": _explainmeetsum_cases,
    "qasper": _qasper_cases,
    "fever": _fever_cases,
    "hotpotqa": _hotpot_cases,
}


def normalize_dataset(
    dataset_id: str,
    *,
    raw_root: Path,
    out_root: Path,
    splits: list[str],
    limit_per_split: int | None,
    seed: int,
) -> NormalizationResult:
    spec = get_dataset_spec(dataset_id)
    dataset_out = out_root / dataset_id
    _ensure_dir(dataset_out)
    if not spec.supports_normalization:
        manifest = {
            "dataset_id": dataset_id,
            "normalized": False,
            "reason": spec.notes,
        }
        manifest_path = dataset_out / "normalization_skipped.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return NormalizationResult(dataset_id=dataset_id, output_path=manifest_path, case_count=0, skipped_reason=spec.notes)

    builder = BUILDERS[dataset_id]
    total_cases = 0
    split_files: list[str] = []
    for offset, split in enumerate(splits):
        cases = builder(raw_root, split=split, limit=limit_per_split, seed=seed + offset)
        out_path = dataset_out / f"{split}.jsonl"
        out_path.write_text("\n".join(case.to_json() for case in cases), encoding="utf-8")
        split_files.append(str(out_path))
        total_cases += len(cases)
    if total_cases == 0:
        raise RuntimeError(
            f"Dataset '{dataset_id}' normalization produced zero executable cases for splits {splits}. "
            "This is treated as a normalization failure, not as success."
        )
    manifest = {
        "dataset_id": dataset_id,
        "normalized": True,
        "splits": split_files,
        "case_count": total_cases,
    }
    manifest_path = dataset_out / "normalization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return NormalizationResult(dataset_id=dataset_id, output_path=manifest_path, case_count=total_cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize downloaded open-scenario datasets into executable case files.")
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset_ids = resolve_dataset_ids([item.strip() for item in args.datasets.split(",") if item.strip()])
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    _ensure_dir(args.out)
    results: list[NormalizationResult] = []
    for dataset_id in dataset_ids:
        try:
            result = normalize_dataset(
                dataset_id,
                raw_root=args.download_root,
                out_root=args.out,
                splits=splits,
                limit_per_split=args.limit_per_split,
                seed=args.seed,
            )
        except Exception as exc:
            dataset_out = args.out / dataset_id
            _ensure_dir(dataset_out)
            manifest = {
                "dataset_id": dataset_id,
                "normalized": False,
                "error_class": type(exc).__name__,
                "error": str(exc),
            }
            manifest_path = dataset_out / "normalization_error.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            result = NormalizationResult(
                dataset_id=dataset_id,
                output_path=manifest_path,
                case_count=0,
                skipped_reason=f"{type(exc).__name__}: {exc}",
                metadata={"error_class": type(exc).__name__},
            )
        results.append(result)
    manifest = {
        "results": [
            {
                "dataset_id": result.dataset_id,
                "output_path": str(result.output_path),
                "case_count": result.case_count,
                "skipped_reason": result.skipped_reason,
            }
            for result in results
        ]
    }
    summary_path = args.out / "normalization_index.json"
    summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"index": str(summary_path), "datasets": [result.dataset_id for result in results]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
