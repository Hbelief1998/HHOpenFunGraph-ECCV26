#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests


# =========================
# 1) 从环境变量读取 DeepSeek Key
# =========================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_TOKEN") or ""

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"  # deepseek-chat / deepseek-reasoner

# 现在仅输出严格 JSON 形式的图谱，不要额外文本，不要 Markdown。
KITCHEN_PROMPT_EN = r"""
You are an expert in household-service robot manipulation and operability modeling.

[Task]
Generate a compact “Kitchen Global Operability Atlas (prior)” for a functional scene graph.
You must output a strict JSON object with:
- "objects": kitchen-relevant O entries with their typical functional carriers (C) and interactive units (U)
- "remote_relation_candidates": directed remote functional candidate edges (E_rem)
  AND each remote edge MUST include a short functional description field "relation".
- "local_function_chains": explicit local functional relation descriptions for O-C, C-U, and O-U (aligned with the hierarchy)

[Definitions]
- O (object): an operability-relevant kitchen entity modeled as a node in the atlas.
  There are TWO kinds of O:
  (A) Directly manipulable objects: must have actionable interfaces (see constraints).
  (B) Remote endpoint objects: may have no local actuators, but can appear as controller/source or target in remote relations
      (e.g., electric outlet, ceiling light). If an object appears in remote_relation_candidates, it MUST appear in "objects".
  To reduce omissions, think by categories (give a few examples after each; examples are not exhaustive):
  • major appliances: refrigerator, oven, microwave, dishwasher
  • small appliances: kettle, coffee machine, rice cooker
  • furniture & storage: cabinet
  • fixtures & plumbing: sink, faucet
  • tabletop containers/tools: bottle, jar
  • infrastructure: ceiling light, switch panel, electric outlet, window, power outlet
- C (functional carrier): a mid-level structure within an object that organizes interaction and typically hosts one or more U
  (e.g., door, drawer, control panel).
- U (interactive unit): an atomic actuator that can be directly pressed/turned/pulled/pushed/toggled
  (e.g., handle, knob, button, cap, lid, switch, stopper, lever).

- Generic category labels are NOT allowed as object names:
  do NOT use vague classes such as "appliance", "home appliance", "device", "equipment", "kitchen utensil", etc.

[Local vs. Remote — representation rule]
- Local relations (E_loc) are part-of control chains inside one entity, typically O <- C <- U or O <- U.
  In this atlas, local relations MUST be expressed ONLY via the O/C/U hierarchy inside each object entry
  (functional_carriers and direct_interactive_units).
  e.g., oven/microwave <- control panel <- knob; cabinet <- drawer/door <- handle/knob; bottle <- cap; kettle/pot/rice cooker <- handle
- Remote relations (E_rem) are cross-entity functional dependencies WITHOUT part-of containment.
  e.g., sink <- faucet, ceiling light <- switch panel.
  Remote relations MUST be expressed ONLY via "remote_relation_candidates". Do NOT mix remote relations into the local hierarchy.

[functional relation descriptions]
- You must ALSO output "local_function_chains" to provide explicit functional descriptions for local relations (O-C, C-U, O-U).
  This does NOT replace the hierarchy; it only adds explicit relation descriptions aligned with it.
- You must ALSO provide a functional description field "relation" for each remote_relation_candidates edge.

=========================
General Constraints (High Priority)
=========================

1) Directly manipulable objects must have actionable interfaces
- For kind (A) objects: include an O only if it has at least one actionable interface:
  (functional_carriers is non-empty) OR (direct_interactive_units is non-empty).
- Exclude entities that are only “grasp-and-move” with no clear actuators.

2) Remote endpoint objects (exception)
- For kind (B) remote endpoint objects (e.g., power outlet, ceiling light):
  they may have empty functional_carriers and empty direct_interactive_units.
  However, include them ONLY if they are used as endpoints in remote_relation_candidates.

3) Minimality (avoid over-generation)
- For each O: prefer 0–2 carriers.
- For each carrier: prefer 1–2 interactive units.
- For direct_interactive_units: prefer 1–3 units.
- If unsure whether something is common, omit it; do not guess long-tail details.

4) Canonical granularity (avoid long-tail parts)
- Prefer high-frequency, generic, generalizable names.
- Avoid rare/over-specific parts (e.g., “freezer door”, “water reservoir lid”, “ice dispenser”, “filter latch”, etc.).
- Do not invent overly fine-grained carrier names: if there is a more standard/simpler carrier, use the simpler one.

5) Naming (stable + compact)
- object / carrier: lowercase English with spaces (no hyphens).
- interactive units: prefer single, generic tokens (e.g., handle, knob, cap, button, switch, lever, stopper) and avoid long or highly specific phrases.
- The examples above are illustrative, not a fixed vocabulary; avoid fine-grained variants such as
  “temperature-knob”, “start-button”, “mode-selector”.

6) U assignment rule (hard rule)
- Each U must be assigned to the smallest structure it directly acts on / directly controls.
- A U may be placed under a carrier’s interactive_units ONLY if both conditions hold:
  (a) the U physically belongs to that carrier (mounted on the carrier or is part of it), AND
  (b) actuating the U directly controls that carrier (open/close, lock/unlock, start/stop, adjust, etc.),
      i.e., the primary effect target of the U is that carrier.
- Otherwise, if the U is used to manipulate the whole object (grasp/carry/pour/move), or it is not specific to a single carrier,
  then it MUST go to the object’s direct_interactive_units.
- U must be an atomic actuator; do NOT use structural/assembly terms as U (e.g., door, drawer, faucet, control panel).

=========================
Remote relation candidates (E_rem only; directed edges)
=========================
Add a top-level field "remote_relation_candidates" listing ONLY remote functional relation candidates (E_rem).

Hard rules:
- Only include plausible remote functional candidates between physically independent entities:
  cross-entity dependencies such as control/trigger/power/enablement.
- Direction MUST be: target <- controller/source.
  In JSON semantics: to_object <- from_object.
- Endpoints must be concrete objects (NOT generic category labels).
- Do NOT include spatial/structural associations such as support, placement, adjacency, containment, layout, or co-location.
  Forbidden examples:
  - cabinet <-> countertop
  - refrigerator/oven/microwave <-> cabinet
  - dishwasher <-> sink
  - switch panel <-> power outlet
- Every endpoint string MUST match exactly one "object" name in the "objects" list.

[Remote functional descriptions (E_rem descriptions)]
- Each remote relation candidate MUST include a short functional description field "relation".
- Use short action phrases, e.g.:
  "provide power", "turn on or off", "control the water flow".

Style examples (not exhaustive):
- ceiling light <- switch panel  (relation: "turn on or off")
- microwave / dishwasher (and other powered appliances) <- electric outlet   (relation: "provide power")
- sink <- faucet                 (relation: "control the water flow")

=========================
Local function chain descriptions (E_loc descriptions; MUST align with hierarchy)
=========================
Add a top-level field "local_function_chains" to describe local functional relations with explicit relation text.

Why: an object can have multiple carriers and multiple units (e.g., microwave has door->handle and control panel->button/knob).
To avoid ambiguity, each chain explicitly binds (object, carrier, unit).

Hard rules:
- Every chain item MUST reference names that appear in "objects" (exact match strings).
- Do NOT invent any extra object/carrier/unit names in local_function_chains.
- For every unit under every carrier in "objects", output one chain item (do not merge).
- If a unit appears under multiple carriers, output one chain per (object, carrier, unit).

Functional description requirements:
- oc_relation (O-C): if hard to describe, you may use "part of".
- cu_relation (C-U) and ou_relation (O-U): MUST be explicit functional action descriptions (non-empty).  If it is hard to distinguish cu_relation and ou_relation, you may write the similar functional descriptions for both.
- Keep relation texts short and action-oriented.

Style examples (not exhaustive):
- cap - bottle: "screw on or off to open or close"
- door - cabinet: "part of"
- handle - door/drawer: "pull to open or close"
- knob - control panel: "rotate to adjust the setting"
- button - control panel: "press to set/start/stop"
- handle - kettle: "grasp to lift or pour"


=========================
Output JSON schema (strict)
=========================
{
  "scene_type": "kitchen",
  "objects": [
    {
      "object": "<object name>",
      "roles": ["O"],
      "functional_carriers": [
        {
          "carrier": "<carrier name>",
          "oc_relation": "<O-C functional description (or 'part of' if unsure)>",
          "interactive_units": [
            {
              "unit": "<unit name>",
              "cu_relation": "<C-U functional description (MUST be non-empty)>",
              "ou_relation": "<O-U functional description (MUST be non-empty)>"
            }
          ]
        }
      ],
      "direct_interactive_units": [
        {
          "unit": "<unit name>",
          "ou_relation": "<O-U functional description (MUST be non-empty)>"
        }
      ]
    }
  ],
  "remote_relation_candidates": [
    {"from_object": "<object>", "to_object": "<object>", "relation": "<functional description>"}
  ]
}

Now output strict JSON only. Do not output any extra text. Do not use Markdown.
""".strip()



def pretty_error(resp: requests.Response) -> str:
    try:
        return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception:
        return (resp.text or "").strip()


def strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        after = stripped[3:]
        newline = after.find("\n")
        stripped = after[newline + 1 :] if newline != -1 else ""
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    s = strip_code_fences(text)
    try:
        return json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i : j + 1])
        raise RuntimeError("Model output is not valid JSON:\n" + (text or ""))


def call_deepseek_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
) -> Dict[str, Any]:
    if not api_key or "PASTE_REAL" in api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY（或 DEEPSEEK_TOKEN）。")

    endpoints = [
        f"{base_url.rstrip('/')}/chat/completions",
        f"{base_url.rstrip('/')}/v1/chat/completions",
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }

    messages = [
        {"role": "system", "content": "Output valid JSON only. Do not include any extra text or Markdown."},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_resp: Optional[requests.Response] = None
    last_exc: Optional[Exception] = None

    for endpoint in endpoints:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                last_resp = resp

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"HTTP {resp.status_code} Unauthorized/Forbidden at {endpoint}\n{pretty_error(resp)}"
                    )

                if resp.status_code == 429 or (500 <= resp.status_code <= 599):
                    if attempt < max_retries:
                        time.sleep(1.5 ** attempt)
                        continue

                raise RuntimeError(f"HTTP {resp.status_code} at {endpoint}\n{pretty_error(resp)}")

            except requests.RequestException as e:
                last_exc = e
                if attempt < max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                break

    if last_resp is not None:
        raise RuntimeError(
            f"DeepSeek request failed. Last HTTP {last_resp.status_code}\n{pretty_error(last_resp)}"
        )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DeepSeek request failed for unknown reason.")


def generate_atlas_with_retry(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    response_retries: int,
    max_tokens_cap: int,
) -> Dict[str, Any]:
    current_max_tokens = max_tokens
    for attempt in range(response_retries + 1):
        resp = call_deepseek_chat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=current_max_tokens,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
        )
        choice = resp["choices"][0]
        content = choice["message"]["content"]
        finish_reason = (choice.get("finish_reason") or "").lower()
        try:
            return extract_json_from_text(content)
        except Exception as exc:
            should_retry = attempt < response_retries
            if finish_reason == "length" and current_max_tokens < max_tokens_cap:
                should_retry = True
                current_max_tokens = min(int(current_max_tokens * 1.5), max_tokens_cap)
            if not should_retry:
                raise
            print(
                f"[deepseek] retry {attempt + 1}/{response_retries} due to parse error ({exc}). "
                f"finish_reason={finish_reason or 'unknown'}, next max_tokens={current_max_tokens}",
                file=sys.stderr,
            )
            time.sleep(min(2.0, 0.5 * (attempt + 1)))

    raise RuntimeError("Failed to obtain valid JSON from DeepSeek after retries.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--response-retries", type=int, default=2,
                        help="Retry parsing/output requests when JSON is truncated")
    parser.add_argument("--max-tokens-cap", type=int, default=4096,
                        help="Upper bound for automatic max_tokens escalation when output is truncated")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    atlas = generate_atlas_with_retry(
        base_url=args.base_url,
        api_key=DEEPSEEK_API_KEY,
        model=args.model,
        user_prompt=KITCHEN_PROMPT_EN,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout,
        max_retries=args.retries,
        response_retries=args.response_retries,
        max_tokens_cap=args.max_tokens_cap,
    )

    print(json.dumps(atlas, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(atlas, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
