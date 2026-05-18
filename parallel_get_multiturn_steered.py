#!/usr/bin/env python3
"""
Multi-turn simulated conversations (user simulator + assistant), with a locally
steered HF assistant (activation steering from a trained probe direction).

User turns: same pipeline as parallel_get_multiturn.py (API or local via make_api_call).
Assistant: Hugging Face causal LM + forward hook from assumption_probes (see sample_and_generate_steered.py).

Examples:
  python parallel_get_multiturn_steered.py data.csv out.csv \\
    meta-llama/Meta-Llama-3.1-8B-Instruct supportv2 \\
    --probe-dir ./probe_out --steer-alpha 1.0 \\
    --user-model gpt4o --max-user-turns 5 --max-workers 1

  # Fixed user persona from prompts.py (turn 2+)
  python parallel_get_multiturn_steered.py data.csv out.csv \\
    Qwen/Qwen2.5-7B-Instruct 4dims \\
    --probe-dir ./probe_out --persona emotional_support
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

# Reuse multi-turn helpers and user-side API/local routing
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

# -----------------------------------------------------------------------------
# Personas (prompts.py) for simulated user
# -----------------------------------------------------------------------------

PERSONA_CHOICES = {
    "validation": support_seeking_user_prompt,
    "support_seeking": support_seeking_user_prompt,
    "objectivity": objectivity_seeking_user_prompt,
    "objectivity_seeking": objectivity_seeking_user_prompt,
    "emotional_support": emotional_support_seeking_user_prompt,
    "emotion_avoidant": emotion_avoidant_user_prompt,
}



def build_simulate_user_prompt(
    history_clean: str,
    seek_validation: Optional[bool] = None,
    persona: Optional[str] = None,
) -> str:
    """Build prompt for simulating User A's next message."""

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


# -----------------------------------------------------------------------------
# Steered assistant (global, guarded by lock for thread safety)
# -----------------------------------------------------------------------------

_assistant_lock = Lock()
_steer_bundle: Optional[dict] = None


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
        raise RuntimeError("Steered assistant already initialized")

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

    direction_np = np.load(os.path.join(probe_dir, "validation_direction.npy")).astype(np.float32)
    direction = torch.tensor(direction_np, device=input_device, dtype=torch.float32)

    best_layer = None
    meta_path = os.path.join(probe_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best_layer = int(meta.get("best_layer", -1))

    layer = layer_override if layer_override >= 0 else (
        best_layer if best_layer is not None and best_layer >= 0 else 20
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


# -----------------------------------------------------------------------------
# Multi-turn row runner (assistant via steered local model)
# -----------------------------------------------------------------------------


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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn sim with steered HF assistant (probe direction hook)."
    )
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument(
        "assistant_model",
        help="Hugging Face model id or path for the steered assistant",
    )
    parser.add_argument("task", help="Assistant prompt type (e.g. supportv2, 4dims)")
    parser.add_argument("--user-model", default=None, help="User simulator model key (default: gpt4o)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--probe-dir", required=True, help="Directory with validation_direction.npy (+ meta.json)")
    parser.add_argument("--steer-alpha", type=float, default=0.0, help="Steering strength (0 = hook no-op)")
    parser.add_argument("--layer", type=int, default=-1, help="Override layer index (-1 = use meta.json)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-impl", default="", help="e.g. flash_attention_2")
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=10)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel rows; assistant generation is locked (default 1 recommended on one GPU).",
    )
    parser.add_argument("--user-sim-mode", type=int, default=1)
    parser.add_argument("--user-sim-switch-turn", type=int, default=5)
    parser.add_argument(
        "--persona",
        default=None,
        choices=sorted(PERSONA_CHOICES.keys()),
        help="If set, use this prompts.py persona for user simulation on turns 2+ "
        "(validation / objectivity switching is disabled).",
    )
    args = parser.parse_args()

    if args.task not in pgm.AVAILABLE_PROMPTS:
        raise ValueError(
            f"Unknown prompt type: {args.task}. Available: {list(pgm.AVAILABLE_PROMPTS.keys())}"
        )

    user_model_key = args.user_model or "gpt4o"
    if user_model_key not in pgm.AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown user model: {user_model_key}. Available: {list(pgm.AVAILABLE_MODELS.keys())}"
        )

    all_records, processed_conv_ids = pgm.load_existing_results(args.output_csv)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    user_client = pgm.initialize_client(user_model_key, args.api_key)
    user_model_name = pgm.AVAILABLE_MODELS[user_model_key]
    print(f"User simulation model: {user_model_key}")

    steer_meta = {
        "steer_alpha": args.steer_alpha,
        "steer_layer": None,
        "assistant_hf_model": args.assistant_model,
        "probe_dir": os.path.abspath(args.probe_dir),
    }

    try:
        _init_steered_assistant(
            hf_model=args.assistant_model,
            probe_dir=args.probe_dir,
            steer_alpha=args.steer_alpha,
            layer_override=args.layer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            use_4bit=args.use_4bit,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            attn_impl=args.attn_impl,
        )
        steer_meta["steer_layer"] = _steer_bundle["layer"]

        df = pd.read_csv(args.input_csv)
        user_cols = pgm.get_sorted_user_cols(df)
        if not user_cols:
            raise ValueError("No columns matching 'user_<n>' or fallback prompt columns in input CSV.")

        if args.sample_n is not None and args.sample_n < len(df):
            df_sub = df.sample(args.sample_n, random_state=42).copy()
        elif args.first_n is not None:
            df_sub = df.head(args.first_n).copy()
        else:
            df_sub = df.copy()

        if "conv_id" in df_sub.columns:
            id_col = "conv_id"
        elif "pair_id" in df_sub.columns:
            id_col = "pair_id"
        else:
            id_col = None
        id_col_output = id_col if id_col else "conv_id"

        if id_col:
            df_remaining = df_sub[~df_sub[id_col].isin(processed_conv_ids)].copy()
        else:
            df_remaining = df_sub[~df_sub.index.isin(processed_conv_ids)].copy()

        if len(df_remaining) == 0:
            print("All conversations already processed.")
            return

        if args.max_workers > 1:
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
                args.task,
                args.max_user_turns,
                args.user_sim_mode,
                args.user_sim_switch_turn,
                args.persona,
                steer_meta,
            )
            for idx, row in df_remaining.iterrows()
        ]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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
                                args.output_csv, index=False
                            )
                        print(
                            f"Progress: saved {len(all_records)} rows "
                            f"(last {id_col_output}={row_id})"
                        )
                except Exception as e:
                    print(f"[ERROR] {id_col_output}={row_id}: {e}")

        pd.DataFrame.from_records(all_records).to_csv(args.output_csv, index=False)
        print(f"Done. Wrote {len(all_records)} rows to {args.output_csv}")
    finally:
        _cleanup_steered_assistant()


if __name__ == "__main__":
    main()
