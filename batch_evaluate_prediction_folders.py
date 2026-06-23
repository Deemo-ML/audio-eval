from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc=None, **_):
            self.iterable = iterable or range(total or 0)
        def __iter__(self):
            return iter(self.iterable)

ARTIFACT_NAMES = ["Clipping", "Hiss", "Buzz", "Pops", "Unnatural Prosody"]
REQUIRED_FILES = ["tfq.jsonl", "mcq.jsonl", "typea_oeq.jsonl", "typeb_oeq.jsonl"]
BASE_OEQ_COLUMNS = {"sample_id", "media_path", "modality", "track_id", "label"}
TCS_WEIGHTS = {"det": 0.4, "hal": 0.3, "perc": 0.3}
SUMMARY_COLUMNS = [
    "Folder",
    "TFQ_Acc",
    "MCQ_Score",
    "TypeB_AccDet",
    "TypeB_Cover",
    "TypeB_CHAIR",
    "TypeB_F0.5",
    "TypeA_Cover",
    "TypeA_CHAIR",
    "TypeA_F0.5",
    "TCS",
]
ARTIFACT_PATTERNS = {
    "Clipping": [r"\bclipping\b", r"\bclipped\b", r"\bdistortion\b", r"\bdistorted\b", r"\bsaturat(?:ed|ion)\b"],
    "Hiss": [r"\bhiss\b", r"\bhissing\b", r"\bhigh[-\s]?frequency\s+(?:static|noise)\b"],
    "Buzz": [r"\bbuzz\b", r"\bbuzzing\b", r"\bhum\b", r"\bhumming\b", r"\blow[-\s]?frequency\s+(?:tone|hum)\b"],
    "Pops": [r"\bpop\b", r"\bpops\b", r"\bclick\b", r"\bclicks\b", r"\babrupt\s+(?:burst|click)\b"],
    "Unnatural Prosody": [r"\bunnatural prosody\b", r"\bprosody\b", r"\brobotic\b", r"\bmonotone\b", r"\bmonotonous\b", r"\bintonation\b", r"\bflat\s+(?:speech|delivery)\b"],
}

PARSER_PROMPT = """You are an OEQ response parser for the TRIDENT audio deepfake challenge.

Your task is to parse the submitted text response only. Do not judge the audio.

Allowed audio artifacts:
- Clipping: waveform saturation, harsh distortion, flattened peaks, clipped or saturated audio.
- Hiss: persistent high-frequency broadband noise or hissing sound.
- Buzz: low-frequency electrical hum, buzzing, humming, or periodic electrical noise.
- Pops: short impulsive clicks, pops, bursts, or abrupt discontinuities.
- Unnatural Prosody: abnormal rhythm, intonation, stress, pauses, monotone delivery, robotic or unnatural speaking style.

Rules:
1. Extract only artifacts explicitly mentioned or clearly described in the response.
2. Do not infer artifacts from the label alone.
3. Do not add artifacts just because they appear in the allowed list.
4. If the response says no clear artifact is detected, return an empty artifact list.
5. Map synonyms to the closest allowed artifact.
6. For Type-B, parse the binary label when present.
7. Return valid JSON only.

Task: {task_name}

Response:
{response}

Return JSON in this exact schema:
{{
  "label": "authentic|manipulated|unknown",
  "artifacts": []
}}
"""


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                records.append(row)
    return records


def load_json_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return read_jsonl(path)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "questions", "annotations", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]
    return []


def normalize_id(value: Any) -> str:
    return str(value or "").strip()


def question_id(record: dict) -> str:
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    return normalize_id(record.get("question_id") or record.get("id") or record.get("qid") or sample.get("question_id"))


def sample_id(record: dict) -> str:
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    return normalize_id(record.get("sample_id") or record.get("audio_id") or record.get("record_id") or sample.get("sample_id") or record.get("id"))


def normalize_modality(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"aud": "audio", "audio": "audio", "vid": "video", "video": "video", "img": "image", "image": "image"}.get(text, text or "unknown")


def answer_value(record: dict) -> Any:
    for key in ("ground_truth", "answer", "answers", "gt_answer", "correct_options", "label", "response"):
        if key in record:
            return record[key]
    return None


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    match = re.search(r"\b(true|false)\b", text)
    if match:
        return match.group(1) == "true"
    return None


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"0", "real", "authentic", "likely authentic"}:
        return "authentic"
    if text in {"1", "fake", "manipulated", "likely manipulated"}:
        return "manipulated"
    return "unknown"


def parse_typeb_label(response: Any) -> str:
    text = str(response or "")
    if "Likely Manipulated" in text:
        return "manipulated"
    if "Likely Authentic" in text:
        return "authentic"
    first = text.splitlines()[0].lower() if text.splitlines() else text.lower()
    if "manipulated" in first or "fake" in first:
        return "manipulated"
    if "authentic" in first or "real" in first:
        return "authentic"
    return "unknown"


def parse_option_set(value: Any, strict: bool = False) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        result = {str(k).strip().upper() for k, v in value.items() if str(v).strip().lower() in {"1", "true", "yes", "y", "on"}}
    elif isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(parse_option_set(item, strict=False))
    else:
        result = {m.upper() for m in re.findall(r"\b[A-E]\b", str(value).upper())}
    result = {x for x in result if re.fullmatch(r"[A-E]", x)}
    if "E" in result and len(result) > 1:
        if strict:
            raise ValueError(f"Invalid option set containing E with other options: {sorted(result)}")
        return {"E"}
    return result


def extract_artifacts_regex(response: Any) -> set[str]:
    text = str(response or "").lower()
    found: set[str] = set()
    for artifact, patterns in ARTIFACT_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            found.add(artifact)
    return found


def parse_artifact_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {artifact for artifact in ARTIFACT_NAMES if str(value.get(artifact, "")).strip().lower() in {"1", "true", "yes", "y", "present", "fake"}}
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        aliases = {a.lower(): a for a in ARTIFACT_NAMES}
        aliases["pop"] = "Pops"
        for item in value:
            canonical = aliases.get(str(item).strip().lower())
            if canonical:
                result.add(canonical)
            else:
                result.update(extract_artifacts_regex(item))
        return result
    return extract_artifacts_regex(value)


def load_choice_questions(data_root: Path, task: str, split: str, modality: str) -> Dict[str, dict]:
    task_dir = data_root / ("MCQ" if task == "mcq" else "TFQ") / split
    rows: Dict[str, dict] = {}
    patterns = [f"{modality[:3]}_*.json", f"{modality}_*.json", "*.json", "*.jsonl"]
    for pattern in patterns:
        for path in sorted(task_dir.glob(pattern)):
            if path.name.startswith("answers"):
                continue
            for row in load_json_records(path):
                qid = question_id(row)
                if qid:
                    rows[qid] = row
    return rows


def infer_option_count(question: dict, default: int = 5) -> int:
    opts = question.get("options") or question.get("choices") or question.get("answers")
    if isinstance(opts, dict):
        return max(default, len(opts))
    if isinstance(opts, list):
        return max(default, len(opts) + (1 if "none of the options are correct" in str(question.get("question", "")).lower() else 0))
    letters = sorted(set(re.findall(r"(?m)^\s*([A-E])\.", str(question.get("question", "")))))
    return max(default, len(letters)) if letters else default


def load_choice_ground_truth(data_root: Path, task: str, split: str, modality: str) -> Dict[str, dict]:
    task_dir = data_root / ("MCQ" if task == "mcq" else "TFQ") / split
    questions = load_choice_questions(data_root, task, split, modality)
    records: List[dict] = []
    for name in ("answers.jsonl", "answers.json"):
        records.extend(load_json_records(task_dir / name))
    if not records:
        records = list(questions.values())
    gt: Dict[str, dict] = {}
    for row in records:
        qid = question_id(row)
        if not qid:
            continue
        qrow = questions.get(qid, row)
        if task == "tfq":
            ans = parse_bool(answer_value(row))
            if ans is None:
                continue
            gt[qid] = {"question_id": qid, "answer": ans, "modality": normalize_modality(qrow.get("modality") or row.get("modality"))}
        else:
            ans_set = parse_option_set(answer_value(row), strict=False)
            if not ans_set and "none of the options are correct" in str(qrow.get("question", "")).lower():
                ans_set = {"E"}
            gt[qid] = {
                "question_id": qid,
                "answer_set": ans_set,
                "option_count": infer_option_count(qrow),
                "modality": normalize_modality(qrow.get("modality") or row.get("modality")),
            }
    return {k: v for k, v in gt.items() if v.get("modality") in {"", "unknown", modality} or modality == "all"}


def parse_truth_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text not in {"", "0", "false", "no", "none", "null", "[]", "[ ]"}


def load_oeq_ground_truth(data_root: Path, split: str, modality: str) -> Dict[str, dict]:
    paths = [
        data_root / "OEQ" / split / f"answers_{modality}.csv",
        data_root / "OEQ" / split / "answers_audio.csv",
        data_root / "OEQ" / split / f"manifest_{modality}.csv",
        data_root / "OEQ" / split / "manifest_audio.csv",
    ]
    gt: Dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            artifact_cols = [c for c in (reader.fieldnames or []) if c not in BASE_OEQ_COLUMNS]
            for row in reader:
                sid = normalize_id(row.get("sample_id"))
                if not sid:
                    continue
                row_modality = normalize_modality(row.get("modality") or modality)
                if modality != "all" and row_modality not in {"", "unknown", modality}:
                    continue
                gt[sid] = {
                    "sample_id": sid,
                    "label": normalize_label(row.get("label") or row.get("authenticity") or row.get("ground_truth")),
                    "modality": row_modality or modality,
                    "artifacts": {a for a in artifact_cols if parse_truth_flag(row.get(a))},
                    "artifact_names": artifact_cols,
                }
        if gt:
            break
    return gt


def score_tfq(records: Sequence[dict], gt: Dict[str, dict]) -> dict:
    pred_by_qid = {question_id(r): r for r in records if question_id(r)}
    correct = 0
    for qid, ans in gt.items():
        pred = pred_by_qid.get(qid)
        if pred is not None and parse_bool(pred.get("response")) == ans["answer"]:
            correct += 1
    total = len(gt)
    return {"acc_tfq": correct / total if total else 0.0, "correct": correct, "total": total}


def score_mcq(records: Sequence[dict], gt: Dict[str, dict]) -> dict:
    pred_by_qid = {question_id(r): r for r in records if question_id(r)}
    total_score = 0.0
    exact = 0
    for qid, ans in gt.items():
        pred_set = parse_option_set(pred_by_qid.get(qid, {}).get("response"), strict=False)
        gt_set = set(ans.get("answer_set") or set()) or {"E"}
        m = int(ans.get("option_count") or 5)
        k = len(gt_set)
        reward = len(pred_set & gt_set) / k if k else 0.0
        penalty = len(pred_set - gt_set) / (m - k) if m > k else 0.0
        total_score += max(0.0, reward - penalty)
        if pred_set == gt_set:
            exact += 1
    total = len(gt)
    return {"score_mcq": total_score / total if total else 0.0, "exact_match_acc": exact / total if total else 0.0, "total": total}


def f_score(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall) if ((beta_sq * precision) + recall) else 0.0


def score_typeb_detection(records: Sequence[dict], gt: Dict[str, dict], parsed: Optional[Dict[str, dict]] = None) -> dict:
    pred_by_sid = {sample_id(r): r for r in records if sample_id(r)}
    correct = 0
    total = 0
    unknown = 0
    for sid, ans in gt.items():
        label = ans.get("label")
        if label not in {"authentic", "manipulated"}:
            continue
        total += 1
        pred_record = pred_by_sid.get(sid)
        if pred_record is None:
            continue
        parsed_label = (parsed or {}).get(sid, {}).get("label")
        pred_label = normalize_label(parsed_label) if parsed_label else parse_typeb_label(pred_record.get("response"))
        if pred_label == "unknown":
            unknown += 1
        if pred_label == label:
            correct += 1
    return {"acc_det": correct / total if total else 0.0, "correct": correct, "total": total, "unknown_label": unknown}


def score_oeq_artifacts(records: Sequence[dict], gt: Dict[str, dict], parsed: Optional[Dict[str, dict]] = None) -> dict:
    pred_by_sid = {sample_id(r): r for r in records if sample_id(r)}
    cover_sum = 0.0
    chair_sum = 0.0
    f_sum = 0.0
    hal_count = 0
    expected_fake = 0
    scored_fake = 0
    for sid, ans in gt.items():
        y_art = set(ans.get("artifacts") or set())
        if not y_art:
            continue
        expected_fake += 1
        pred = pred_by_sid.get(sid)
        if pred is None:
            chair_sum += 1.0
            hal_count += 1
            continue
        scored_fake += 1
        if parsed is not None and sid in parsed:
            r_art = set(parsed[sid].get("artifacts") or [])
        else:
            r_art = extract_artifacts_regex(pred.get("response"))
        tp = len(r_art & y_art)
        cover = tp / max(1, len(y_art))
        chair = 1.0 if not r_art else 1.0 - (tp / len(r_art))
        precision = 1.0 - chair
        f05 = f_score(precision, cover, beta=0.5)
        cover_sum += cover
        chair_sum += chair
        f_sum += f05
        if chair > 0:
            hal_count += 1
    return {
        "cover": cover_sum / expected_fake if expected_fake else 0.0,
        "chair": chair_sum / expected_fake if expected_fake else 0.0,
        "hal_rate": hal_count / expected_fake if expected_fake else 0.0,
        "f_0_5": f_sum / expected_fake if expected_fake else 0.0,
        "expected_fake_samples": expected_fake,
        "fake_samples_scored": scored_fake,
    }


def parser_cache_key(task_name: str, response: str) -> str:
    payload = json.dumps([task_name, response], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_parser_cache(path: Optional[Path]) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                cache[str(row["key"])] = row
    return cache


def save_parser_cache(path: Optional[Path], cache: Dict[str, dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key in sorted(cache):
            f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Optional[dict]:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_parsed(value: Any, task_name: str, response: str) -> dict:
    if not isinstance(value, dict):
        return {"label": parse_typeb_label(response) if task_name == "typeb_oeq" else "unknown", "artifacts": sorted(extract_artifacts_regex(response))}
    return {
        "label": normalize_label(value.get("label")),
        "artifacts": [a for a in ARTIFACT_NAMES if a in parse_artifact_set(value.get("artifacts"))],
    }


class QwenOEQParser:
    def __init__(
        self,
        model_path: Path,
        cache_path: Optional[Path],
        batch_size: int,
        max_new_tokens: int,
        device: str,
        dtype: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Qwen parser requires torch and transformers.") from exc
        self.torch = torch
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = int(max_new_tokens)
        self.cache_path = cache_path
        self.cache = load_parser_cache(cache_path)
        self.device = device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
        torch_dtype = self._resolve_dtype(dtype)
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_dtype(self, dtype: str) -> Any:
        text = str(dtype or "auto").lower()
        if self.device == "cpu" and text in {"bf16", "bfloat16", "fp16", "float16"}:
            text = "float32"
        return {
            "auto": None,
            "bf16": self.torch.bfloat16,
            "bfloat16": self.torch.bfloat16,
            "fp16": self.torch.float16,
            "float16": self.torch.float16,
            "fp32": self.torch.float32,
            "float32": self.torch.float32,
        }.get(text)

    def parse_records(self, task_name: str, records: Sequence[dict]) -> Dict[str, dict]:
        parsed: Dict[str, dict] = {}
        pending: List[Tuple[str, str, str]] = []
        for row in records:
            sid = sample_id(row)
            if not sid:
                continue
            response = str(row.get("response", ""))
            key = parser_cache_key(task_name, response)
            cached = self.cache.get(key)
            if cached is not None:
                parsed[sid] = normalize_parsed(cached.get("parsed"), task_name, response)
            else:
                pending.append((sid, response, key))
        for start in tqdm(range(0, len(pending), self.batch_size), desc=f"Qwen parse {task_name}"):
            batch = pending[start : start + self.batch_size]
            outputs = self._generate_batch(task_name, [response for _, response, _ in batch])
            for (sid, response, key), output in zip(batch, outputs):
                obj = extract_json_object(output)
                item = normalize_parsed(obj, task_name, response)
                self.cache[key] = {"key": key, "task_name": task_name, "response": response, "parsed": item}
                parsed[sid] = item
        return parsed

    def _generate_batch(self, task_name: str, responses: Sequence[str]) -> List[str]:
        prompts = [PARSER_PROMPT.format(task_name=task_name, response=response) for response in responses]
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception:
            return [json.dumps(normalize_parsed(None, task_name, response), ensure_ascii=False) for response in responses]

    def save(self) -> None:
        save_parser_cache(self.cache_path, self.cache)


def has_submission_files(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).exists() for name in REQUIRED_FILES)


def discover_submission_folders(root: Path, recursive: bool) -> List[Path]:
    if has_submission_files(root):
        return [root]
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted([p for p in iterator if has_submission_files(p)], key=lambda p: str(p).lower())


def safe_token(path: Path, root: Path) -> str:
    try:
        text = str(path.relative_to(root))
    except ValueError:
        text = path.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.replace("/", "__").replace("\\", "__")).strip("_") or path.name


def evaluate_folder(folder: Path, data_root: Path, split: str, modality: str, parser: Optional[QwenOEQParser], oeq_parser: str, tcs_hal_source: str) -> Dict[str, Any]:
    records = {name[:-6]: read_jsonl(folder / name) for name in REQUIRED_FILES}
    tfq_gt = load_choice_ground_truth(data_root, "tfq", split, modality)
    mcq_gt = load_choice_ground_truth(data_root, "mcq", split, modality)
    oeq_gt = load_oeq_ground_truth(data_root, split, modality)

    parsed_typea: Optional[Dict[str, dict]] = None
    parsed_typeb: Optional[Dict[str, dict]] = None
    if oeq_parser == "qwen":
        if parser is None:
            raise RuntimeError("Internal error: Qwen parser is not initialized.")
        parsed_typea = parser.parse_records("typea_oeq", records["typea_oeq"])
        parsed_typeb = parser.parse_records("typeb_oeq", records["typeb_oeq"])

    tfq = score_tfq(records["tfq"], tfq_gt)
    mcq = score_mcq(records["mcq"], mcq_gt)
    typeb_det = score_typeb_detection(records["typeb_oeq"], oeq_gt, parsed_typeb)
    typea_art = score_oeq_artifacts(records["typea_oeq"], oeq_gt, parsed_typea)
    typeb_art = score_oeq_artifacts(records["typeb_oeq"], oeq_gt, parsed_typeb)

    hal_f05 = typea_art["f_0_5"] if tcs_hal_source == "typea" else (typea_art["f_0_5"] + typeb_art["f_0_5"]) / 2.0
    s_det = 100.0 * typeb_det["acc_det"]
    s_perc = 100.0 * (0.5 * tfq["acc_tfq"] + 0.5 * mcq["score_mcq"])
    s_hal = 100.0 * hal_f05
    tcs = TCS_WEIGHTS["det"] * s_det + TCS_WEIGHTS["hal"] * s_hal + TCS_WEIGHTS["perc"] * s_perc

    return {
        "folder": str(folder),
        "tfq": tfq,
        "mcq": mcq,
        "typeb_oeq": {**typeb_det, "cover": typeb_art["cover"], "chair": typeb_art["chair"], "f_0_5": typeb_art["f_0_5"]},
        "typea_oeq": {"cover": typea_art["cover"], "chair": typea_art["chair"], "f_0_5": typea_art["f_0_5"]},
        "tcs": {"S_Det": s_det, "S_Perc": s_perc, "S_Hal": s_hal, "TCS": tcs, "hal_source": tcs_hal_source},
    }


def row_from_summary(folder: Path, root: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    try:
        name = str(folder.relative_to(root)).replace("\\", "/")
    except ValueError:
        name = folder.name
    return {
        "Folder": name,
        "TFQ_Acc": float(summary.get("tfq", {}).get("acc_tfq", 0.0)),
        "MCQ_Score": float(summary.get("mcq", {}).get("score_mcq", 0.0)),
        "TypeB_AccDet": float(summary.get("typeb_oeq", {}).get("acc_det", 0.0)),
        "TypeB_Cover": float(summary.get("typeb_oeq", {}).get("cover", 0.0)),
        "TypeB_CHAIR": float(summary.get("typeb_oeq", {}).get("chair", 0.0)),
        "TypeB_F0.5": float(summary.get("typeb_oeq", {}).get("f_0_5", 0.0)),
        "TypeA_Cover": float(summary.get("typea_oeq", {}).get("cover", 0.0)),
        "TypeA_CHAIR": float(summary.get("typea_oeq", {}).get("chair", 0.0)),
        "TypeA_F0.5": float(summary.get("typea_oeq", {}).get("f_0_5", 0.0)),
        "TCS": float(summary.get("tcs", {}).get("TCS", 0.0)),
    }


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    widths = {col: len(col) for col in SUMMARY_COLUMNS}
    formatted: List[Dict[str, str]] = []
    for row in rows:
        item: Dict[str, str] = {}
        for col in SUMMARY_COLUMNS:
            value = row.get(col, "")
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            item[col] = text
            widths[col] = max(widths[col], len(text))
        formatted.append(item)
    print("  ".join(col.ljust(widths[col]) for col in SUMMARY_COLUMNS))
    print("  ".join("-" * widths[col] for col in SUMMARY_COLUMNS))
    for row in formatted:
        print("  ".join(row[col].ljust(widths[col]) for col in SUMMARY_COLUMNS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate flat TRIDENT audio submission folders with optional Qwen3.5-4B OEQ parsing.")
    parser.add_argument("--predictions-root", "--root-dir", dest="predictions_root", type=Path, required=True, help="Folder containing submission subfolders, or one submission folder.")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/trident"))
    parser.add_argument("--split", default="public_val", choices=["train", "public_val"])
    parser.add_argument("--modality", default="audio", choices=["audio", "image", "video", "all"])
    parser.add_argument("--output-csv", type=Path, default=None, help="CSV table path. Defaults to <predictions-root>/batch_scorer_summary.csv.")
    parser.add_argument("--score-json-dir", type=Path, default=None, help="Per-folder JSON output dir. Defaults to <predictions-root>/evaluation_results/folder_scores.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oeq-parser", choices=["qwen", "regex"], default="qwen")
    parser.add_argument("--qwen-model-path", type=Path, default=Path("/root/Qwen3.5-4B"), help="Local Qwen3.5-4B path for OEQ response parsing.")
    parser.add_argument("--parser-cache-dir", type=Path, default=None, help="Cache dir for Qwen parser outputs. Defaults to <predictions-root>/evaluation_results/parser_cache.")
    parser.add_argument("--parser-batch-size", type=int, default=32)
    parser.add_argument("--parser-max-new-tokens", type=int, default=128)
    parser.add_argument("--parser-device", default="cuda")
    parser.add_argument("--parser-dtype", default="bfloat16")
    parser.add_argument("--tcs-hal-source", choices=["mean", "typea"], default="mean", help="Use mean(Type-A, Type-B) or Type-A F0.5 as S_Hal.")
    parser.add_argument("--sort-by", choices=SUMMARY_COLUMNS, default="TCS")
    parser.add_argument("--ascending", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.predictions_root.resolve()
    folders = discover_submission_folders(root, recursive=args.recursive)
    if not folders:
        raise SystemExit(f"No valid submission folders found under {root}. Required files: {', '.join(REQUIRED_FILES)}")

    output_csv = args.output_csv or (root / "batch_scorer_summary.csv")
    score_json_dir = args.score_json_dir or (root / "evaluation_results" / "folder_scores")
    cache_dir = args.parser_cache_dir or (root / "evaluation_results" / "parser_cache")
    score_json_dir.mkdir(parents=True, exist_ok=True)
    if args.oeq_parser == "qwen":
        cache_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, folder in enumerate(folders, start=1):
        print(f"\n[{idx}/{len(folders)}] Evaluating {folder}", flush=True)
        token = safe_token(folder, root)
        score_json = score_json_dir / f"{token}.score_summary.json"
        try:
            if args.reuse_existing and score_json.exists():
                summary = json.loads(score_json.read_text(encoding="utf-8"))
            else:
                parser_obj: Optional[QwenOEQParser] = None
                if args.oeq_parser == "qwen":
                    parser_obj = QwenOEQParser(
                        model_path=args.qwen_model_path,
                        cache_path=cache_dir / f"{token}.oeq_parser_cache.jsonl",
                        batch_size=args.parser_batch_size,
                        max_new_tokens=args.parser_max_new_tokens,
                        device=args.parser_device,
                        dtype=args.parser_dtype,
                    )
                summary = evaluate_folder(
                    folder,
                    args.data_root,
                    args.split,
                    args.modality,
                    parser_obj,
                    args.oeq_parser,
                    args.tcs_hal_source,
                )
                if parser_obj is not None:
                    parser_obj.save()
                score_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rows.append(row_from_summary(folder, root, summary))
        except Exception as exc:
            message = str(exc)
            print(f"ERROR: {folder}: {message}", file=sys.stderr)
            errors.append({"Folder": str(folder), "Error": message})
            if not args.continue_on_error:
                raise

    if args.sort_by == "Folder":
        rows.sort(key=lambda x: str(x.get("Folder", "")), reverse=not args.ascending)
    else:
        rows.sort(key=lambda x: float(x.get(args.sort_by, 0.0)), reverse=not args.ascending)
    write_summary_csv(output_csv, rows)
    print(f"\nWrote CSV summary: {output_csv}")
    print_table(rows)

    if errors:
        error_csv = output_csv.with_name(output_csv.stem + "_errors.csv")
        with error_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Folder", "Error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\n{len(errors)} folder(s) failed. Error report: {error_csv}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
