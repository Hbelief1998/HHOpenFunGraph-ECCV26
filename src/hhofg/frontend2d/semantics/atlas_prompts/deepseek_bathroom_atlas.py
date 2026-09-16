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

# 仅输出严格 JSON，不要额外文本，不要 Markdown。
BATHROOM_PROMPT_EN = r"""
You are an expert in household-service robot manipulation and operability modeling.

[Task]
Generate a compact “Bathroom Global Operability Atlas (prior)” for a hierarchical functional scene graph.
You must output a strict JSON object containing:

- "objects": bathroom-relevant O entries with their typical functional carriers (C) and interactive units (U).
- "remote_relation_candidates": directed remote functional candidate edges (E_rem).
  Each remote edge MUST include a short functional description field "relation".
- Local functional relationship descriptions MUST be written inline in the O/C/U hierarchy:
  carrier.oc_relation, unit.cu_relation, unit.ou_relation, and direct_unit.ou_relation.

[Definitions]

- O (object): an operability-relevant bathroom entity modeled as a node in the atlas.
  There are TWO kinds of O:
  (A) Directly manipulable objects: must have actionable interfaces.
  (B) Remote endpoint/source objects: may have no local actuators, but can appear as controller/source or target in remote relations
  (e.g., ceiling light, exhaust fan, power outlet). If an object appears in remote_relation_candidates, it MUST appear in "objects".

  To reduce omissions, think by categories. Examples are not exhaustive:
  • major fixtures: toilet, sink, bathtub, shower
  • plumbing/control fixtures: faucet, shower valve
  • furniture & storage: bathroom cabinet, vanity cabinet
  • hygiene containers/tools: bottle, jar, trash can
  • room infrastructure: ceiling light, switch panel, power outlet, door, window

- C (functional carrier): a mid-level structure within an object that organizes interaction and typically hosts one or more U
  (e.g., tank, door, drawer, lid, control panel).
- U (interactive unit): an atomic actuator that can be directly pressed/turned/pulled/pushed/toggled/grasped
  (e.g., handle, knob, button, lever, switch, cap, lid, stopper, pump, latch, pedal).

- Generic category labels are NOT allowed as object names:
  do NOT use vague classes such as "fixture", "bathroom fixture", "appliance", "device", "equipment", "container", "tool", etc.

[Local vs. Remote — representation rule]

- Local relations (E_loc) are part-of or internal control chains inside one entity, typically O <- C <- U or O <- U.
  Local relations MUST be expressed ONLY via the O/C/U hierarchy inside each object entry:
  "functional_carriers" and "direct_interactive_units".
  Examples:
  - toilet <- tank <- lever/button
  - bathroom cabinet <- drawer/door <- handle/knob
  - bottle <- cap
  - trash can <- lid <- pedal, or trash can <- pedal

- Remote relations (E_rem) are cross-entity functional dependencies WITHOUT part-of containment.
  Remote relations MUST be expressed ONLY via "remote_relation_candidates".
  Examples:
  - ceiling light <- switch panel
  - sink <- faucet
  - bathtub <- faucet
  - shower <- shower valve

=========================
General Constraints (High Priority)
=========================

1. Directly manipulable objects must have actionable interfaces

- For kind (A) objects: include an O only if it has at least one actionable interface:
  functional_carriers is non-empty OR direct_interactive_units is non-empty.
- Exclude entities that are only “grasp-and-move” with no clear actuator or functional interface.
- Examples usually to exclude, unless a clear actuator is visible/annotated:
  towel, towel rack, mirror, drain, mat, soap bar, toothbrush holder, shelf, wall, floor, ceiling.

2. Remote endpoint/source objects exception

- For kind (B) remote endpoint/source objects, e.g., ceiling light, power outlet:
  they may have empty functional_carriers and empty direct_interactive_units.
  However, include them ONLY if they are used as endpoints in remote_relation_candidates.

3. Minimality

- For each O: prefer 0–2 carriers.
- For each carrier: prefer 1–2 interactive units.
- For direct_interactive_units: prefer 1–3 units.
- If unsure whether something is common, omit it.
- Do not guess long-tail or rare parts.

4. Canonical granularity

- Prefer high-frequency, generic, generalizable names.
- Avoid rare or overly specific parts such as "overflow drain cover", "filter latch", "hidden release", "water reservoir lid".
- Do not invent overly fine-grained carrier names when a simpler carrier is sufficient.

5. Naming

- object / carrier / unit names MUST be lowercase English with spaces.
- Do not use hyphens.
- interactive units should be short generic labels:
  handle, knob, button, lever, switch, cap, lid, stopper, pump, latch, pedal.
- Avoid fine-grained variants such as "flush-button", "temperature-knob", "water-flow-lever", "power-switch".
  Use "button", "knob", "lever", or "switch" instead.

6. C validity rule

- C must be a structural carrier, not an atomic actuator.
- Allowed C examples: door, drawer, tank, lid, control panel.
- Forbidden C labels: handle, knob, button, lever, switch, cap, stopper, pump, latch, pedal, drain.
- A carrier MUST NOT have the same name as one of its interactive units.
- For simple actuator objects, put the actuator in direct_interactive_units instead of functional_carriers:
  faucet <- handle/lever/knob, shower valve <- handle/lever/knob, bottle <- cap, switch panel <- switch,
  sink <- stopper, bathtub <- stopper.

7. U assignment rule

- Each U must be assigned to the smallest structure it directly acts on or directly controls.
- A U may be placed under a carrier’s interactive_units ONLY if both conditions hold:
  (a) the U physically belongs to that carrier or is mounted on that carrier; AND
  (b) actuating the U directly controls that carrier, e.g., open/close, lock/unlock, flush, dispense, adjust.
- Otherwise, if the U is used to manipulate the whole object, grasp/carry/move it, or is not specific to a single carrier,
  then it MUST go to the object’s direct_interactive_units.
- U must be an atomic actuator.
  Do NOT use structural terms as U:
  door, drawer, tank, faucet, showerhead, shower valve, control panel, sink, bathtub, toilet.

=========================
Remote relation candidates (E_rem only; directed edges)
=========================

Add a top-level field "remote_relation_candidates" listing ONLY remote functional relation candidates.

Hard rules:

- Only include plausible remote functional candidates between physically independent entities:
  cross-entity dependencies such as control, trigger, power, water flow, drainage enablement.
- Direction MUST be: target <- controller/source.
  In JSON semantics: to_object <- from_object.
- from_object MUST be the controller/source.
- to_object MUST be the controlled/consumer/affected target.
- Endpoints must be concrete objects, NOT generic category labels.
- Every endpoint string MUST match exactly one "object" name in the "objects" list.
- Do NOT include local part-of relations here.
- Do NOT include purely spatial/structural associations such as support, placement, adjacency, containment, layout, or co-location.
  Forbidden examples:
  - toilet <-> sink
  - bathtub <-> shower
  - cabinet <-> sink
  - mirror <-> sink
  - switch panel <-> power outlet
  - showerhead <-> shower
- Keep the remote list compact.

Recommended bathroom remote patterns:

- ceiling light <- switch panel relation: "turn on or off"
- sink <- faucet relation: "control the water flow"
- bathtub <- faucet relation: "fill with water"
- shower <- shower valve relation: "control the water flow"

Remote functional description requirements:

- Each remote relation candidate MUST include a short non-empty "relation".
- Use short action phrases:
  "turn on or off", "provide power", "control the water flow", "fill with water".

=========================
Local relation descriptions
=========================

Local functional relationship descriptions MUST be written inline:

- For each carrier C under each object O, include:
  "oc_relation": a short O-C description. If unsure, use "part of".
- For each unit U under each carrier C, use:
  {"unit": "...", "cu_relation": "...", "ou_relation": "..."}
  cu_relation and ou_relation MUST be non-empty.
- For each direct unit U under each object O, use:
  {"unit": "...", "ou_relation": "..."}
  ou_relation MUST be non-empty.

Style examples:

- tank - toilet: "part of"
- lever - tank: "push or pull to flush"
- handle - door/drawer: "pull to open or close"
- knob - control panel: "rotate to adjust the setting"
- button - control panel: "press to start or stop"
- cap - bottle: "twist to open or close"
- stopper - sink/bathtub: "open or close the drain"
- pedal - trash can: "press to open the lid"

=========================
Output JSON schema (strict)
=========================

{
"scene_type": "bathroom",
"atlas_version": "bathroom_v2_ocu_relation_text",
"objects": [
{
"object": "<object name>",
"roles": ["O"],
"functional_carriers": [
{
"carrier": "<carrier name>",
"oc_relation": "<O-C description, or 'part of'>",
"interactive_units": [
{
"unit": "<unit name>",
"cu_relation": "<C-U functional description, non-empty>",
"ou_relation": "<O-U functional description, non-empty>"
}
]
}
],
"direct_interactive_units": [
{
"unit": "<unit name>",
"ou_relation": "<O-U functional description, non-empty>"
}
]
}
],
"remote_relation_candidates": [
{
"from_object": "<controller/source object>",
"to_object": "<controlled/target object>",
"relation": "<functional description, non-empty>"
}
]
}

Now output strict JSON only.
Do not output any extra text.
Do not use Markdown.
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
    if not api_key or api_key.startswith("sk-REPLACE"):
        raise RuntimeError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY (or DEEPSEEK_TOKEN).")

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
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--response-retries",
        type=int,
        default=2,
        help="Retry parsing/output requests when JSON is truncated",
    )
    parser.add_argument(
        "--max-tokens-cap",
        type=int,
        default=4096,
        help="Upper bound for automatic max_tokens escalation when output is truncated",
    )
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    atlas = generate_atlas_with_retry(
        base_url=args.base_url,
        api_key=DEEPSEEK_API_KEY,
        model=args.model,
        user_prompt=BATHROOM_PROMPT_EN,
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
