#!/usr/bin/env python
"""
Multi-turn simulated conversations (user simulator + assistant), with a locally
steered HF assistant (activation steering from a trained probe direction).

User turns: same pipeline as parallel_get_multiturn.py (API or local via make_api_call).
Assistant: Hugging Face causal LM + forward hook from assumption_probes.

Single run (one alpha, one setting):
  python parallel_get_multiturn_steered.py data.csv out.csv \\
    meta-llama/Llama-3.3-70B-Instruct 4dims \\
    --probe-dir ./probe_out --steer-alpha 1.0 --use-4bit

Batch run (one model load; all alphas × all settings):
  python parallel_get_multiturn_steered.py --batch \\
    --output-dir test_results/objectivity_seeking_steered \\
    --steer-alphas -1,-0.5,0.5 \\
    --setting data/valpairs-modified-obj-15.csv,valpairs_obj_m2_sw11,2,11 \\
    --setting data/valpairs-modified-obj-15.csv,valpairs_obj_m2_sw5,2,5 \\
    --setting data/valpairs-modified-val-15.csv,valpairs_val_m1_sw5,1,5 \\
    --setting data/valpairs-modified-val-15.csv,valpairs_val_m1_sw11,1,11 \\
    meta-llama/Llama-3.3-70B-Instruct 4dims \\
    --probe-dir ./probe_out --use-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

import parallel_get_multiturn as pgm

from prompts import (
    emotion_avoidant_user_prompt,
    emotional_support_seeking_user_prompt,
    objectivity_seeking_user_prompt,
    support_seeking_user_prompt,
    user_llm_system_prompt,
)

_PROBES_DIR = os.path.join(os.path.dirname(__file__), "verbalizedassumptions", "assumption_probes")
if _PROBES_DIR not in sys.path:
    sys.path.insert(0, _PROBES_DIR)

import sample_and_generate_steered as sgs  # noqa: E402

PERSONA_CHOICES = {
    "validation": support_seeking_user_prompt,
    "support_seeking": support_seeking_user_prompt,
    "objectivity": objectivity_seeking_user_prompt,
    "objectivity_seeking": objectivity_seeking_user_prompt,
    "emotional_support": emotional_support_seeking_user_prompt,
    "emotion_avoidant": emotion_avoidant_user_prompt,
}

DEFAULT_SETTINGS = [
    ("data/valpairs-modified-obj-15.csv", "valpairs_obj_m2_sw11", 2, 11),
    ("data/valpairs-modified-obj-15.csv", "valpairs_obj_m2_sw5", 2, 5),
    ("data/valpairs-modified-val-15.csv", "valpairs_val_m1_sw5", 1, 5),
    ("data/valpairs-modified-val-15.csv", "valpairs_val_m1_sw11", 1, 11),
]


@dataclass(frozen=True)
class RunSetting:
    input_csv: str
    out_prefix: str
    user_sim_mode: int
    user_sim_switch_turn: int


def build_simulate_user_prompt(
    history_clean: str,
    seek_validation: Optional[bool] = None,
    persona: Optional[str] = None,
) -> str:
    base = (
        f"{user_llm_system_prompt}\n"
        f"Conversation so far:\n"
        f'"""\n'
        f"{history_clean}\n"
        f'"""\n\n'
        f'Generate only what User A would say next—one or a few natural messages, '
        f'as a real user would respond. Output nothing else: no labels, '
        f'no "User A:", no explanation. Just the next user message.'
    )

    if persona:
        snippet = PERSONA_CHOICES.get(persona)
        if snippet is None:
            raise ValueError(
                f"Unknown persona {persona!r}. Choose from: {sorted(PERSONA_CHOICES.keys())}"
            )
        base += snippet
    else:
        if seek_validation is True:
            base += support_seeking_user_prompt
        elif seek_validation is False:
            base += objectivity_seeking_user_prompt
    return base


_assistant_lock = Lock()
_steer_bundle: Optional[dict] = None


def parse_alphas(alpha_str: str) -> list[float]:
    return [float(x.strip()) for x in alpha_str.split(",") if x.strip()]


def parse_setting(spec: str) -> RunSetting:
    """Parse input_csv,out_prefix,sim_mode,switch_turn."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --setting {spec!r}; expected "
            "input_csv,out_prefix,sim_mode,switch_turn"
        )
    return RunSetting(parts[0], parts[1], int(parts[2]), int(parts[3]))


def alpha_subdir_name(alpha: float) -> str:
    return f"alpha_{alpha}"


def alpha_to_flat_suffix(alpha: float) -> str:
    if alpha == -1:
        return "alpha_m1"
    if alpha == -0.5:
        return "alpha_m0p5"
    if alpha == 0.5:
        return "alpha0p5"
    if alpha == 1:
        return "alpha1"
    if alpha == 0:
        return "base"
    return "alpha_" + str(alpha).replace(".", "p")


def resolve_output_csv(
    output_dir: str,
    layout: str,
    alpha: float,
    out_prefix: str,
) -> str:
    if layout == "alpha_subdir":
        return os.path.join(output_dir, alpha_subdir_name(alpha), f"{out_prefix}.csv")
    if layout == "flat_suffix":
        suffix = alpha_to_flat_suffix(alpha)
        return os.path.join(output_dir, f"{out_prefix}_{suffix}.csv")
    raise ValueError(f"Unknown output layout: {layout!r}")


def set_steer_alpha(alpha: float) -> None:
    if _steer_bundle is None:
        raise RuntimeError("Steered assistant not initialized")
    _steer_bundle["hook_state"]["alpha"] = float(alpha)
    _steer_bundle["steer_alpha"] = float(alpha)


def _init_steered_assistant(
    hf_model: str,
    probe_dir: str,
    steer_alpha: float,
    layer_override: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_4bit: bool,
    device_map: str,
    trust_remote_code: bool,
    attn_impl: str,
) -> None:
    global _steer_bundle
    if _steer_bundle is not None:
        set_steer_alpha(steer_alpha)
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ns = SimpleNamespace(
        model=hf_model,
        trust_remote_code=trust_remote_code,
        use_4bit=use_4bit,
        device_map=device_map,
        attn_impl=attn_impl or "",
    )
    model, tokenizer, input_device = sgs.build_model_and_tokenizer(ns)

    direction_np = np.load(os.path.join(probe_dir, "validation_direction.npy")).astype(
        np.float32
    )
    direction = torch.tensor(direction_np, device=input_device, dtype=torch.float32)

    best_layer = None
    meta_path = os.path.join(probe_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best_layer = int(meta.get("best_layer", -1))

    layer = (
        layer_override
        if layer_override >= 0
        else (best_layer if best_layer is not None and best_layer >= 0 else 20)
    )

    hook_handle, hook_state = sgs.register_single_alpha_hook(model, layer, direction)
    hook_state["alpha"] = float(steer_alpha)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    _steer_bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "input_device": input_device,
        "gen_kwargs": gen_kwargs,
        "hook_handle": hook_handle,
        "hook_state": hook_state,
        "layer": layer,
        "steer_alpha": float(steer_alpha),
        "probe_dir": probe_dir,
        "hf_model": hf_model,
    }
    print(
        f"Steered assistant ready: model={hf_model} layer={layer} alpha={steer_alpha} "
        f"probe_dir={probe_dir}"
    )


def _steered_assistant_generate(prompt: str) -> str:
    if _steer_bundle is None:
        raise RuntimeError("Steered assistant not initialized")
    model = _steer_bundle["model"]
    tokenizer = _steer_bundle["tokenizer"]
    input_device = _steer_bundle["input_device"]
    gen_kwargs = _steer_bundle["gen_kwargs"]

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    with _assistant_lock:
        inputs = tokenizer(input_text, return_tensors="pt").to(input_device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def _cleanup_steered_assistant() -> None:
    global _steer_bundle
    if _steer_bundle is not None:
        _steer_bundle["hook_handle"].remove()
        _steer_bundle = None


def run_for_row_steered(
    user_client,
    user_model_name: str,
    row: pd.Series,
    user_cols: list[str],
    prompt_type: str,
    max_user_turns: Optional[int],
    user_sim_mode: Optional[int],
    user_sim_switch_turn: int,
    persona: Optional[str],
    assistant_generate: Callable[[str], str],
    steer_meta: dict,
) -> list[dict]:
    history_str = ""
    if "history" in row.index:
        history_str = row["history"] or ""

    if not user_cols:
        return []
    first_col = user_cols[0]
    if first_col not in row.index:
        return []
    initial_user_text = row[first_col]
    if not isinstance(initial_user_text, str) or not initial_user_text.strip():
        return []
    initial_user_text = initial_user_text.strip()

    num_turns = max_user_turns if max_user_turns is not None else 10
    history_clean = ""
    outputs = []

    for turn_idx in range(1, num_turns + 1):
        if turn_idx == 1:
            user_text = initial_user_text
            user_simulated = False
        else:
            if persona:
                sim_prompt = build_simulate_user_prompt(history_clean, persona=persona)
                user_text = pgm.make_api_call(user_client, user_model_name, sim_prompt)
                user_text = (user_text or "").strip()
                user_simulated = True
            else:
                seek_validation = None
                if user_sim_mode == 1:
                    seek_validation = turn_idx <= user_sim_switch_turn
                elif user_sim_mode == 2:
                    seek_validation = turn_idx > user_sim_switch_turn

                sim_prompt = build_simulate_user_prompt(
                    history_clean, seek_validation=seek_validation
                )
                user_text = pgm.make_api_call(user_client, user_model_name, sim_prompt)
                user_text = (user_text or "").strip()
                if seek_validation:
                    user_text = f"{user_text} I am explicitly seeking validation."
                else:
                    user_text = f"{user_text} I am explicitly seeking objectivity."
                user_simulated = True

            if not user_text or user_text == "ERROR":
                break

        prompt = pgm.build_prompt_with_history(history_str, user_text, prompt_type)
        model_output = assistant_generate(prompt)
        assistant_reply = pgm.extract_assistant_reply(model_output)

        rec = {
            "user_turn_index": turn_idx,
            "user_col": f"user_{turn_idx}",
            "user_text": user_text,
            "model_output": model_output,
            "user_simulated": user_simulated,
            "user_sim_mode": user_sim_mode,
            "user_sim_switch_turn": user_sim_switch_turn,
            "persona": persona or "",
            **steer_meta,
        }
        outputs.append(rec)

        if history_str:
            history_str += "\n"
            history_clean += "\n"
        history_clean += f"User: {user_text}\nAI: {assistant_reply}"
        history_str = history_clean

    return outputs


def process_row_wrapper(args):
    (
        row_id,
        id_col,
        row,
        user_client,
        user_model_name,
        user_cols,
        prompt_type,
        max_user_turns,
        user_sim_mode,
        user_sim_switch_turn,
        persona,
        steer_meta,
    ) = args
    try:
        per_row = run_for_row_steered(
            user_client=user_client,
            user_model_name=user_model_name,
            row=row,
            user_cols=user_cols,
            prompt_type=prompt_type,
            max_user_turns=max_user_turns,
            user_sim_mode=user_sim_mode,
            user_sim_switch_turn=user_sim_switch_turn,
            persona=persona,
            assistant_generate=_steered_assistant_generate,
            steer_meta=steer_meta,
        )
        records = []
        metadata = {}
        for col in row.index:
            if not re.match(r"user_\d+$", col):
                metadata[col] = row[col]
        for o in per_row:
            records.append({id_col: row_id, **metadata, **o})
        return row_id, records, None
    except Exception as e:
        print(f"[ERROR] Failed processing {id_col}={row_id}: {e}")
        return row_id, [], str(e)


def _prepare_dataframe(
    input_csv: str,
    sample_n: Optional[int],
    first_n: Optional[int],
) -> tuple[pd.DataFrame, list[str], Optional[str], str]:
    df = pd.read_csv(input_csv)
    user_cols = pgm.get_sorted_user_cols(df)
    if not user_cols:
        raise ValueError(
            "No columns matching 'user_<n>' or fallback prompt columns in input CSV."
        )

    if sample_n is not None and sample_n < len(df):
        df_sub = df.sample(sample_n, random_state=42).copy()
    elif first_n is not None:
        df_sub = df.head(first_n).copy()
    else:
        df_sub = df.copy()

    if "conv_id" in df_sub.columns:
        id_col = "conv_id"
    elif "pair_id" in df_sub.columns:
        id_col = "pair_id"
    else:
        id_col = None
    id_col_output = id_col if id_col else "conv_id"
    return df_sub, user_cols, id_col, id_col_output


def _get_processed_conv_ids(output_csv: str) -> set:
    """Read pair_id/conv_id from an existing output file (no console noise)."""
    if not os.path.exists(output_csv):
        return set()
    try:
        existing_df = pd.read_csv(output_csv)
    except Exception:
        return set()
    if "conv_id" in existing_df.columns:
        return set(existing_df["conv_id"].unique())
    if "pair_id" in existing_df.columns:
        return set(existing_df["pair_id"].unique())
    return set()


def count_remaining_conversations(
    input_csv: str,
    output_csv: str,
    sample_n: Optional[int],
    first_n: Optional[int],
) -> int:
    processed_conv_ids = _get_processed_conv_ids(output_csv)
    df_sub, _, id_col, _ = _prepare_dataframe(input_csv, sample_n, first_n)
    if id_col:
        df_remaining = df_sub[~df_sub[id_col].isin(processed_conv_ids)].copy()
    else:
        df_remaining = df_sub[~df_sub.index.isin(processed_conv_ids)].copy()
    return len(df_remaining)


def is_configuration_complete(
    input_csv: str,
    output_csv: str,
    sample_n: Optional[int],
    first_n: Optional[int],
) -> bool:
    """True when every conversation in the input subset is already in output_csv."""
    return count_remaining_conversations(input_csv, output_csv, sample_n, first_n) == 0


def run_one_configuration(
    *,
    input_csv: str,
    output_csv: str,
    assistant_model: str,
    task: str,
    probe_dir: str,
    steer_alpha: float,
    user_client,
    user_model_name: str,
    layer_override: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_4bit: bool,
    device_map: str,
    trust_remote_code: bool,
    attn_impl: str,
    sample_n: Optional[int],
    first_n: Optional[int],
    max_user_turns: int,
    max_workers: int,
    user_sim_mode: int,
    user_sim_switch_turn: int,
    persona: Optional[str],
) -> bool:
    """
    Run conversations for one (setting, alpha) pair.
    Returns True if any work was done, False if skipped (already complete).
    """
    if is_configuration_complete(input_csv, output_csv, sample_n, first_n):
        print(f"SKIP (already complete): {output_csv}")
        return False

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    all_records, processed_conv_ids = pgm.load_existing_results(output_csv)

    steer_meta = {
        "steer_alpha": steer_alpha,
        "steer_layer": None,
        "assistant_hf_model": assistant_model,
        "probe_dir": os.path.abspath(probe_dir),
    }

    _init_steered_assistant(
        hf_model=assistant_model,
        probe_dir=probe_dir,
        steer_alpha=steer_alpha,
        layer_override=layer_override,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_4bit=use_4bit,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_impl=attn_impl,
    )
    steer_meta["steer_layer"] = _steer_bundle["layer"]

    df_sub, user_cols, id_col, id_col_output = _prepare_dataframe(
        input_csv, sample_n, first_n
    )

    if id_col:
        df_remaining = df_sub[~df_sub[id_col].isin(processed_conv_ids)].copy()
    else:
        df_remaining = df_sub[~df_sub.index.isin(processed_conv_ids)].copy()

    print(f"\n{'=' * 80}")
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"steer_alpha={steer_alpha}  user_sim_mode={user_sim_mode}  "
          f"switch_turn={user_sim_switch_turn}")
    print(f"Remaining conversations: {len(df_remaining)} / {len(df_sub)}")
    print(f"{'=' * 80}\n")

    if max_workers > 1:
        print(
            "Note: max-workers>1 only helps if user simulator is remote; "
            "assistant shares one GPU model behind a lock."
        )

    save_lock = Lock()
    tasks = [
        (
            row[id_col] if id_col else idx,
            id_col_output,
            row,
            user_client,
            user_model_name,
            user_cols,
            task,
            max_user_turns,
            user_sim_mode,
            user_sim_switch_turn,
            persona,
            steer_meta,
        )
        for idx, row in df_remaining.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row_wrapper, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            row_id = futures[fut]
            try:
                _rid, records, err = fut.result()
                if err:
                    print(f"[WARNING] {id_col_output}={row_id}: {err}")
                else:
                    all_records.extend(records)
                    with save_lock:
                        pd.DataFrame.from_records(all_records).to_csv(
                            output_csv, index=False
                        )
                    print(
                        f"Progress: saved {len(all_records)} rows "
                        f"(last {id_col_output}={row_id})"
                    )
            except Exception as e:
                print(f"[ERROR] {id_col_output}={row_id}: {e}")

    pd.DataFrame.from_records(all_records).to_csv(output_csv, index=False)
    print(f"Done. Wrote {len(all_records)} rows to {output_csv}")
    return True


def run_batch(args: argparse.Namespace, settings: list[RunSetting]) -> None:
    alphas = parse_alphas(args.steer_alphas)
    if not alphas:
        raise ValueError("No alphas in --steer-alphas")

    jobs: list[tuple[float, RunSetting, str]] = []
    for alpha in alphas:
        for setting in settings:
            out_csv = resolve_output_csv(
                args.output_dir, args.output_layout, alpha, setting.out_prefix
            )
            jobs.append((alpha, setting, out_csv))

    pending: list[tuple[float, RunSetting, str]] = []
    skipped = 0
    print(f"Batch plan: {len(alphas)} alphas × {len(settings)} settings = {len(jobs)} outputs")
    for alpha, setting, out_csv in jobs:
        if is_configuration_complete(
            setting.input_csv, out_csv, args.sample_n, args.first_n
        ):
            skipped += 1
            print(f"  SKIP  alpha={alpha} {setting.out_prefix} -> {out_csv}")
        else:
            pending.append((alpha, setting, out_csv))
            n_left = count_remaining_conversations(
                setting.input_csv, out_csv, args.sample_n, args.first_n
            )
            print(f"  RUN   alpha={alpha} {setting.out_prefix} -> {out_csv} ({n_left} convs left)")

    print(f"Skipping {skipped} complete output(s); running {len(pending)} job(s).")
    if not pending:
        print("All batch outputs already complete — no model load.")
        return

    user_model_key = args.user_model or "gpt4o"
    if user_model_key not in pgm.AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown user model: {user_model_key}. Available: {list(pgm.AVAILABLE_MODELS.keys())}"
        )
    user_client = pgm.initialize_client(user_model_key, args.api_key)
    user_model_name = pgm.AVAILABLE_MODELS[user_model_key]
    print(f"User simulation model: {user_model_key}")

    try:
        for alpha, setting, out_csv in pending:
            print(f"\n>>> alpha={alpha} setting={setting.out_prefix}")
            run_one_configuration(
                input_csv=setting.input_csv,
                output_csv=out_csv,
                assistant_model=args.assistant_model,
                task=args.task,
                probe_dir=args.probe_dir,
                steer_alpha=alpha,
                user_client=user_client,
                user_model_name=user_model_name,
                layer_override=args.layer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                use_4bit=args.use_4bit,
                device_map=args.device_map,
                trust_remote_code=args.trust_remote_code,
                attn_impl=args.attn_impl,
                sample_n=args.sample_n,
                first_n=args.first_n,
                max_user_turns=args.max_user_turns,
                max_workers=args.max_workers,
                user_sim_mode=setting.user_sim_mode,
                user_sim_switch_turn=setting.user_sim_switch_turn,
                persona=args.persona,
            )
    finally:
        _cleanup_steered_assistant()

    print(f"\nBatch finished. Outputs under {args.output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-turn sim with steered HF assistant (probe direction hook)."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="One model load; loop alphas and settings. Skips any output CSV that "
        "already contains all conversations (requires --output-dir, --steer-alphas).",
    )
    parser.add_argument("input_csv", nargs="?", help="Input CSV (single-run mode)")
    parser.add_argument("output_csv", nargs="?", help="Output CSV (single-run mode)")
    parser.add_argument(
        "assistant_model",
        help="Hugging Face model id or path for the steered assistant",
    )
    parser.add_argument("task", help="Assistant prompt type (e.g. supportv2, 4dims)")
    parser.add_argument("--user-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument(
        "--steer-alpha",
        type=float,
        default=None,
        help="Steering strength for single-run mode (default 0)",
    )
    parser.add_argument(
        "--steer-alphas",
        default="0",
        help="Comma-separated alphas for --batch (e.g. -1,-0.5,0,0.5,1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root for --batch (writes alpha_*/prefix.csv or flat suffixes)",
    )
    parser.add_argument(
        "--output-layout",
        choices=("alpha_subdir", "flat_suffix"),
        default="alpha_subdir",
        help="alpha_subdir: out_dir/alpha_X/prefix.csv; flat_suffix: out_dir/prefix_suffix.csv",
    )
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="SPEC",
        help="Batch setting: input_csv,out_prefix,sim_mode,switch_turn (repeatable)",
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--user-sim-mode", type=int, default=1)
    parser.add_argument("--user-sim-switch-turn", type=int, default=5)
    parser.add_argument("--persona", default=None, choices=sorted(PERSONA_CHOICES.keys()))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task not in pgm.AVAILABLE_PROMPTS:
        raise ValueError(
            f"Unknown prompt type: {args.task}. Available: {list(pgm.AVAILABLE_PROMPTS.keys())}"
        )

    if args.batch:
        if not args.output_dir:
            raise ValueError("--batch requires --output-dir")
        settings = [parse_setting(s) for s in args.setting] if args.setting else [
            RunSetting(*t) for t in DEFAULT_SETTINGS
        ]
        os.makedirs(args.output_dir, exist_ok=True)
        run_batch(args, settings)
        return

    if not args.input_csv or not args.output_csv:
        parser.error("single-run mode requires input_csv and output_csv (or use --batch)")

    steer_alpha = 0.0 if args.steer_alpha is None else args.steer_alpha

    if is_configuration_complete(
        args.input_csv, args.output_csv, args.sample_n, args.first_n
    ):
        print(f"SKIP (already complete): {args.output_csv}")
        return

    user_model_key = args.user_model or "gpt4o"
    if user_model_key not in pgm.AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown user model: {user_model_key}. Available: {list(pgm.AVAILABLE_MODELS.keys())}"
        )
    user_client = pgm.initialize_client(user_model_key, args.api_key)
    user_model_name = pgm.AVAILABLE_MODELS[user_model_key]
    print(f"User simulation model: {user_model_key}")

    try:
        run_one_configuration(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            assistant_model=args.assistant_model,
            task=args.task,
            probe_dir=args.probe_dir,
            steer_alpha=steer_alpha,
            user_client=user_client,
            user_model_name=user_model_name,
            layer_override=args.layer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            use_4bit=args.use_4bit,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            attn_impl=args.attn_impl,
            sample_n=args.sample_n,
            first_n=args.first_n,
            max_user_turns=args.max_user_turns,
            max_workers=args.max_workers,
            user_sim_mode=args.user_sim_mode,
            user_sim_switch_turn=args.user_sim_switch_turn,
            persona=args.persona,
        )
    finally:
        _cleanup_steered_assistant()


if __name__ == "__main__":
    main()
