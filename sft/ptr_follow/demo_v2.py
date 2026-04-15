"""Interactive demo for the dual-AR anchor-then-expand model on ptr_follow.

Visualises the step-by-step generation process in a Tkinter window (X11).

Generation order:
  1. Anchor: find max-confidence position across both models, reveal it.
  2. L2R:    fill positions right of anchor, left to right.
  3. R2L:    fill positions left of anchor, right to left.

Usage:
    python sft/ptr_follow/demo_v2.py --model 170 --ckpt_path <path> --order reverse
    python sft/ptr_follow/demo_v2.py --model 170 --l2r_path <l2r.pth> --r2l_path <r2l.pth> --order reverse
"""

import re
import sys
import argparse
import tkinter as tk
from tkinter import scrolledtext, font as tkfont
from pathlib import Path
from functools import partial

# ── path setup ──────────────────────────────────────────────────────────────
_here = Path(__file__).parent.resolve()      # sft/ptr_follow/
_sft  = _here.parent                         # sft/          (ptr_follow_data.py lives here)
_root = _sft.parent                          # repo root     (lit_gpt/ lives here)
for p in [str(_root), str(_sft)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file

from lit_gpt.model import GPT, Config


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     type=int, required=True, help="Model size (e.g. 170)")
    p.add_argument("--order",     type=str, default="reverse", help="reverse | middle")
    p.add_argument("--num_val",   type=int, default=100,   help="Validation samples to draw from")
    p.add_argument("--ckpt_path", type=str, default="",    help="Combined dual-AR checkpoint (.pth)")
    p.add_argument("--l2r_path",  type=str, default="",    help="L2R model checkpoint")
    p.add_argument("--r2l_path",  type=str, default="",    help="R2L model checkpoint")
    args = p.parse_args()
    if not args.ckpt_path and not (args.l2r_path and args.r2l_path):
        p.error("Provide --ckpt_path or both --l2r_path and --r2l_path")
    return args


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(path, config, device, state_key=None):
    model = GPT(config)
    model.apply(partial(model._init_weights, n_layer=config.n_layer))
    if path.endswith(".safetensors"):
        sd = load_file(path, device="cpu")
    else:
        raw = torch.load(path, map_location="cpu")
        if state_key is not None and state_key in raw:
            sd = raw[state_key]
        elif "model" in raw:
            sd = raw["model"]
        else:
            sd = raw
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"Loaded {'(' + state_key + ')' if state_key else ''} from {path}")
    return model


# ── data loading ──────────────────────────────────────────────────────────────

def load_val_samples(tokenizer, order, num_val,
                     max_prompt_length=256, max_response_length=16):
    """Load the last num_val samples from the HF dataset."""
    file_path = f"zzy1123/ptr_follow_{order}_order_sft"
    dataset = load_dataset(file_path, split="train")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    raw = []
    for data in dataset:
        question_text = data["input"]
        answer_text   = data["output"].replace("\n", " ")

        q_ids = tokenizer(question_text, return_tensors="pt")["input_ids"][0]
        q_len = q_ids.shape[-1]
        if q_len > max_prompt_length:
            continue
        q_pad = torch.full((max_prompt_length - q_len,), pad_id, dtype=q_ids.dtype)
        q_ids = torch.cat([q_pad, q_ids])          # left-padded to 256

        a_ids = tokenizer(answer_text, return_tensors="pt")["input_ids"][0]
        if a_ids[0] == tokenizer.bos_token_id:
            a_ids = a_ids[1:]
        a_len = a_ids.shape[-1]
        if a_len > max_response_length:
            continue
        a_pad = torch.full((max_response_length - a_len,), tokenizer.eos_token_id, dtype=a_ids.dtype)
        a_ids = torch.cat([a_ids, a_pad])           # right-padded to 16

        raw.append(dict(
            prompt_ids=q_ids,
            target_ids=a_ids,
            question_text=question_text,
            answer_text=answer_text,
        ))

    val_samples = raw[-num_val:]
    print(f"Loaded {len(val_samples)} validation samples.")
    if val_samples:
        print("[DEBUG] First question_text:\n", val_samples[0]["question_text"][:300])
    return val_samples


# ── prompt parsing ────────────────────────────────────────────────────────────

_MAPPING_RE = re.compile(
    r'^([a-zA-Z])\s*(?:->|-->|→|:)\s*([^\s,;]+)',
    re.IGNORECASE,
)
_INLINE_RE = re.compile(r'([a-zA-Z])\s*(?:->|-->|→|:)\s*([^\s,;]+)', re.IGNORECASE)


def parse_prompt(text):
    """Return (mappings: List[(key_lower, val)], query_lines: List[str])."""
    mappings, query_lines = [], []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _MAPPING_RE.match(line)
        if m:
            mappings.append((m.group(1).lower(), m.group(2)))
        else:
            inline = _INLINE_RE.findall(line)
            if len(inline) > 1:
                for k, v in inline:
                    mappings.append((k.lower(), v))
            else:
                query_lines.append(line)
    return mappings, query_lines


# ── generation: capture all steps (anchor-then-expand) ───────────────────────

MASK_TOKEN_ID = 0
RESPONSE_LENGTH = 16


@torch.no_grad()
def capture_all_steps(model_l2r, model_r2l, prompt_ids, tokenizer, device):
    """Run anchor-then-expand and return a list of per-step state dicts.

    Step ordering:
      steps[0]          → initial state (all masked); predictions from both models
      steps[1]          → anchor token revealed (source = "anchor")
      steps[2..]        → L2R fills right of anchor, left to right (source = "l2r")
      steps[2+r2l_count] → R2L fills left of anchor, right to left (source = "r2l")

    Each entry:
        response   : List[int]    token ids (0 = masked)
        filled     : List[bool]
        best_pos   : int          position just filled (-1 for step 0)
        token_str  : str          decoded token just placed ("" for step 0)
        source     : str          "anchor" | "l2r" | "r2l" | ""
        confidence : float
        l2r_preds  : List[int]    top-1 token from L2R at each position
        r2l_preds  : List[int]    top-1 token from R2L at each position
        l2r_confs  : List[float]  L2R confidence at each position
        r2l_confs  : List[float]  R2L confidence at each position
    """
    prompt   = prompt_ids.unsqueeze(0).to(device)   # (1, prompt_len)
    resp_len = RESPONSE_LENGTH

    response = torch.full((1, resp_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
    filled   = torch.zeros(1, resp_len, dtype=torch.bool, device=device)

    use_autocast = device.type == "cuda"

    def _run_models():
        x         = torch.cat([prompt, response],                dim=-1)
        flipped_x = torch.cat([prompt, response.flip(dims=[1])], dim=-1)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                l2r_logits = model_l2r(x)        [:, -(resp_len + 1):-1]
                r2l_logits = model_r2l(flipped_x)[:, -(resp_len + 1):-1].flip(dims=[1])
        else:
            l2r_logits = model_l2r(x)        [:, -(resp_len + 1):-1]
            r2l_logits = model_r2l(flipped_x)[:, -(resp_len + 1):-1].flip(dims=[1])

        l2r_probs = torch.softmax(l2r_logits.float(), dim=-1)
        r2l_probs = torch.softmax(r2l_logits.float(), dim=-1)
        cl2r, pl2r = torch.max(l2r_probs, dim=-1)
        cr2l, pr2l = torch.max(r2l_probs, dim=-1)
        return cl2r, pl2r, cr2l, pr2l

    def _state(bp=-1, tok_str="", src="", conf=0.0,
               cl2r=None, pl2r=None, cr2l=None, pr2l=None):
        _zi = [0]   * resp_len
        _zf = [0.0] * resp_len
        return dict(
            response   = response[0].cpu().tolist(),
            filled     = filled[0].cpu().tolist(),
            best_pos   = bp,
            token_str  = tok_str,
            source     = src,
            confidence = conf,
            l2r_preds  = pl2r[0].cpu().tolist() if pl2r is not None else list(_zi),
            r2l_preds  = pr2l[0].cpu().tolist() if pr2l is not None else list(_zi),
            l2r_confs  = cl2r[0].cpu().tolist() if cl2r is not None else list(_zf),
            r2l_confs  = cr2l[0].cpu().tolist() if cr2l is not None else list(_zf),
        )

    # ── Step 0: run both models on fully-masked state ─────────────────────────
    cl2r, pl2r, cr2l, pr2l = _run_models()
    steps = [_state(cl2r=cl2r, pl2r=pl2r, cr2l=cr2l, pr2l=pr2l)]

    # ── Phase 1: find and reveal anchor ───────────────────────────────────────
    # Anchor position determined by combined (element-wise max) confidence
    combined_conf = torch.max(cl2r, cr2l)   # (1, resp_len)
    anchor_pos = torch.argmax(combined_conf, dim=-1)[0].item()
    # pick token and conf from whichever model was more confident at anchor_pos
    if cl2r[0, anchor_pos].item() >= cr2l[0, anchor_pos].item():
        anchor_tok  = pl2r[0, anchor_pos].item()
        anchor_conf = cl2r[0, anchor_pos].item()
    else:
        anchor_tok  = pr2l[0, anchor_pos].item()
        anchor_conf = cr2l[0, anchor_pos].item()

    response[0, anchor_pos] = anchor_tok
    filled[0, anchor_pos]   = True
    tok_str = tokenizer.decode([anchor_tok]) if anchor_tok != MASK_TOKEN_ID else "[MASK]"

    cl2r, pl2r, cr2l, pr2l = _run_models()
    steps.append(_state(
        bp=anchor_pos, tok_str=tok_str, src="anchor",
        conf=anchor_conf, cl2r=cl2r, pl2r=pl2r, cr2l=cr2l, pr2l=pr2l,
    ))

    # ── Phase 2: L2R expansion — positions right of anchor ───────────────────
    for j in range(resp_len):
        if j <= anchor_pos or filled[0, j].item():
            continue

        # greedy L2R prediction at position j given current response state
        x = torch.cat([prompt, response], dim=-1)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model_l2r(x)[:, -(resp_len + 1):-1]
        else:
            logits = model_l2r(x)[:, -(resp_len + 1):-1]

        probs = torch.softmax(logits[0, j].float(), dim=-1)
        conf, tok = probs.max(dim=-1)

        response[0, j] = tok.item()
        filled[0, j]   = True
        tok_str = tokenizer.decode([tok.item()]) if tok.item() != MASK_TOKEN_ID else "[MASK]"

        cl2r, pl2r, cr2l, pr2l = _run_models()
        steps.append(_state(
            bp=j, tok_str=tok_str, src="l2r", conf=conf.item(),
            cl2r=cl2r, pl2r=pl2r, cr2l=cr2l, pr2l=pr2l,
        ))

    # ── Phase 3: R2L expansion — positions left of anchor ────────────────────
    for j in range(resp_len - 1, -1, -1):
        if j >= anchor_pos or filled[0, j].item():
            continue

        flipped_x = torch.cat([prompt, response.flip(dims=[1])], dim=-1)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                r2l_logits = model_r2l(flipped_x)[:, -(resp_len + 1):-1].flip(dims=[1])
        else:
            r2l_logits = model_r2l(flipped_x)[:, -(resp_len + 1):-1].flip(dims=[1])

        probs = torch.softmax(r2l_logits[0, j].float(), dim=-1)
        conf, tok = probs.max(dim=-1)

        response[0, j] = tok.item()
        filled[0, j]   = True
        tok_str = tokenizer.decode([tok.item()]) if tok.item() != MASK_TOKEN_ID else "[MASK]"

        cl2r, pl2r, cr2l, pr2l = _run_models()
        steps.append(_state(
            bp=j, tok_str=tok_str, src="r2l", conf=conf.item(),
            cl2r=cl2r, pl2r=pl2r, cr2l=cr2l, pr2l=pr2l,
        ))

    return steps


# ── Tkinter Demo GUI ──────────────────────────────────────────────────────────

class DemoApp:
    AUTO_PLAY_MS = 600

    C_BG        = "#1e1e2e"
    C_PANEL     = "#2a2a3e"
    C_PANEL_ALT = "#313248"
    C_TEXT      = "#cdd6f4"
    C_MUTED     = "#6c7086"
    C_MASKED    = "#45475a"
    C_FILLED    = "#a6e3a1"
    C_ANCHOR    = "#cba6f7"   # purple — anchor position
    C_JUST_L2R  = "#f9e2af"   # gold   — just filled by L2R
    C_JUST_R2L  = "#89dceb"   # cyan   — just filled by R2L
    C_CORRECT   = "#a6e3a1"
    C_WRONG     = "#f38ba8"
    C_BORDER    = "#89b4fa"
    C_BTN       = "#313244"
    C_BTN_ACT   = "#89dceb"
    C_KEY       = "#89b4fa"
    C_ARROW     = "#6c7086"
    C_VAL       = "#a6e3a1"

    def __init__(self, root, val_samples, model_l2r, model_r2l, tokenizer, device):
        self.root        = root
        self.val_samples = val_samples
        self.model_l2r   = model_l2r
        self.model_r2l   = model_r2l
        self.tokenizer   = tokenizer
        self.device      = device

        self.sample_idx   = 0
        self.current_step = 0
        self.steps        = []
        self._after_id    = None
        self._anchor_pos  = -1   # remembered for colour coding

        self._build_ui()
        self._load_sample(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root
        r.title("Dual-AR Anchor-Then-Expand Demo")
        r.configure(bg=self.C_BG)
        r.resizable(True, True)

        mono      = tkfont.Font(family="Courier New", size=10)
        mono_sm   = tkfont.Font(family="Courier New", size=9)
        mono_lg   = tkfont.Font(family="Courier New", size=13, weight="bold")
        sans      = tkfont.Font(family="Helvetica", size=11)
        sans_bold = tkfont.Font(family="Helvetica", size=11, weight="bold")
        sans_sm   = tkfont.Font(family="Helvetica", size=9,  weight="bold")

        pad = dict(padx=10, pady=4)

        # ── sample header ─────────────────────────────────────────────────────
        hdr_frame = tk.Frame(r, bg=self.C_PANEL, relief="flat")
        hdr_frame.pack(fill="x", **pad)
        self.lbl_sample = tk.Label(
            hdr_frame, text="", bg=self.C_PANEL, fg=self.C_BORDER,
            font=sans_bold, anchor="w",
        )
        self.lbl_sample.pack(side="left", padx=6, pady=4)

        # ── prompt area ───────────────────────────────────────────────────────
        tk.Label(r, text="PROMPT", bg=self.C_BG, fg=self.C_MUTED,
                 font=sans_bold, anchor="w").pack(fill="x", padx=10)

        prompt_frame = tk.Frame(r, bg=self.C_BG)
        prompt_frame.pack(fill="x", padx=10, pady=(0, 6))

        map_outer = tk.Frame(prompt_frame, bg=self.C_PANEL, relief="flat",
                             bd=1, highlightthickness=1,
                             highlightbackground=self.C_BORDER)
        map_outer.pack(side="left", fill="y", padx=(0, 6))

        tk.Label(map_outer, text="MAPPING RULES", bg=self.C_PANEL,
                 fg=self.C_MUTED, font=sans_sm, anchor="w").pack(
            fill="x", padx=4, pady=(2, 0))

        map_grid = tk.Frame(map_outer, bg=self.C_PANEL)
        map_grid.pack(padx=4, pady=(0, 4))

        self._map_val_labels  = {}
        self._map_row_frames  = {}
        for col_start, grid_col in [(0, 0), (13, 2)]:
            for row_i in range(13):
                letter = chr(ord('a') + col_start + row_i)
                bg = self.C_PANEL if (col_start + row_i) % 2 == 0 else self.C_PANEL_ALT
                row_f = tk.Frame(map_grid, bg=bg)
                row_f.grid(row=row_i, column=grid_col, padx=(0, 10), pady=0, sticky="w")
                self._map_row_frames[letter] = row_f

                tk.Label(row_f, text=letter, width=2,
                         bg=bg, fg=self.C_KEY, font=mono, anchor="e").pack(side="left")
                tk.Label(row_f, text="->", bg=bg, fg=self.C_ARROW,
                         font=mono).pack(side="left")
                val_lbl = tk.Label(row_f, text="?", width=5,
                                   bg=bg, fg=self.C_VAL, font=mono, anchor="w")
                val_lbl.pack(side="left")
                self._map_val_labels[letter] = val_lbl

        query_frame = tk.Frame(prompt_frame, bg=self.C_BG)
        query_frame.pack(side="left", fill="both", expand=True)

        tk.Label(query_frame, text="QUERY", bg=self.C_BG,
                 fg=self.C_MUTED, font=sans_sm, anchor="w").pack(fill="x")
        self.txt_query = scrolledtext.ScrolledText(
            query_frame, height=6, bg=self.C_PANEL, fg=self.C_TEXT,
            insertbackground=self.C_TEXT, font=mono,
            relief="flat", wrap="word", state="disabled",
        )
        self.txt_query.pack(fill="both", expand=True)

        # ── step info ─────────────────────────────────────────────────────────
        info_frame = tk.Frame(r, bg=self.C_PANEL, relief="flat")
        info_frame.pack(fill="x", padx=10, pady=4)
        self.lbl_step = tk.Label(
            info_frame, text="", bg=self.C_PANEL, fg=self.C_TEXT,
            font=sans_bold, anchor="w",
        )
        self.lbl_step.pack(side="left", padx=6, pady=4)

        # ── token grid ────────────────────────────────────────────────────────
        grid_container = tk.Frame(r, bg=self.C_BG)
        grid_container.pack(fill="x", padx=10, pady=4)

        tk.Label(r, text="L2R PREDICTIONS", bg=self.C_BG, fg=self.C_MUTED,
                 font=sans_bold, anchor="w").pack(fill="x", padx=10)
        self.l2r_labels = []
        for i in range(RESPONSE_LENGTH):
            lbl = tk.Label(
                grid_container, text="—\n—", width=7, relief="groove",
                bg=self.C_PANEL, fg=self.C_MUTED, font=mono_sm,
                padx=2, pady=2, justify="center",
            )
            lbl.grid(row=0, column=i, padx=2, pady=1)
            self.l2r_labels.append(lbl)

        tk.Label(r, text="R2L PREDICTIONS", bg=self.C_BG, fg=self.C_MUTED,
                 font=sans_bold, anchor="w").pack(fill="x", padx=10)
        self.r2l_labels = []
        for i in range(RESPONSE_LENGTH):
            lbl = tk.Label(
                grid_container, text="—\n—", width=7, relief="groove",
                bg=self.C_PANEL, fg=self.C_MUTED, font=mono_sm,
                padx=2, pady=2, justify="center",
            )
            lbl.grid(row=1, column=i, padx=2, pady=1)
            self.r2l_labels.append(lbl)

        tk.Label(r, text="RESPONSE", bg=self.C_BG, fg=self.C_MUTED,
                 font=sans_bold, anchor="w").pack(fill="x", padx=10)
        self.resp_labels = []
        for i in range(RESPONSE_LENGTH):
            lbl = tk.Label(
                grid_container, text="[MASK]", width=7, relief="groove",
                bg=self.C_MASKED, fg="white", font=mono_lg,
                padx=2, pady=4,
            )
            lbl.grid(row=2, column=i, padx=2, pady=2)
            self.resp_labels.append(lbl)

        tk.Label(r, text="TARGET", bg=self.C_BG, fg=self.C_MUTED,
                 font=sans_bold, anchor="w").pack(fill="x", padx=10)
        self.tgt_labels = []
        for i in range(RESPONSE_LENGTH):
            lbl = tk.Label(
                grid_container, text="?", width=7, relief="groove",
                bg=self.C_PANEL, fg=self.C_TEXT, font=mono_lg,
                padx=2, pady=4,
            )
            lbl.grid(row=3, column=i, padx=2, pady=2)
            self.tgt_labels.append(lbl)

        # ── accuracy label ────────────────────────────────────────────────────
        self.lbl_acc = tk.Label(r, text="", bg=self.C_BG, fg=self.C_TEXT, font=sans_bold)
        self.lbl_acc.pack(fill="x", padx=10, pady=(0, 4))

        # ── control buttons ───────────────────────────────────────────────────
        btn_frame = tk.Frame(r, bg=self.C_BG)
        btn_frame.pack(pady=8)

        btn_style = dict(font=sans_bold, relief="flat", padx=14, pady=6, cursor="hand2")

        self.btn_prev = tk.Button(
            btn_frame, text="<  Prev", bg=self.C_BTN, fg=self.C_TEXT,
            command=self._prev_step, **btn_style,
        )
        self.btn_prev.pack(side="left", padx=4)

        self.btn_next = tk.Button(
            btn_frame, text="Next  >", bg=self.C_BTN, fg=self.C_TEXT,
            command=self._next_step, **btn_style,
        )
        self.btn_next.pack(side="left", padx=4)

        self.btn_play = tk.Button(
            btn_frame, text=">>  Auto Play", bg=self.C_BTN, fg=self.C_BTN_ACT,
            command=self._toggle_play, **btn_style,
        )
        self.btn_play.pack(side="left", padx=4)

        self.btn_new = tk.Button(
            btn_frame, text="[New] Sample", bg=self.C_BTN, fg=self.C_TEXT,
            command=self._new_sample, **btn_style,
        )
        self.btn_new.pack(side="left", padx=4)

    # ── sample management ─────────────────────────────────────────────────────

    def _load_sample(self, idx):
        self._stop_play()
        self.sample_idx = idx % len(self.val_samples)
        sample = self.val_samples[self.sample_idx]

        prompt_text = sample["question_text"]
        mappings, query_lines = parse_prompt(prompt_text)

        mapping_dict = {k: v for k, v in mappings}
        for letter, val_lbl in self._map_val_labels.items():
            val = mapping_dict.get(letter, "?")
            val_lbl.configure(text=val)

        self.txt_query.configure(state="normal")
        self.txt_query.delete("1.0", "end")
        self.txt_query.insert("end", "\n".join(query_lines))
        self.txt_query.configure(state="disabled")

        target_ids = sample["target_ids"].tolist()
        eos_id = self.tokenizer.eos_token_id
        for i, tid in enumerate(target_ids):
            tok = self.tokenizer.decode([tid]) if tid != eos_id else "<eos>"
            self.tgt_labels[i].configure(text=tok.strip() or "·",
                                         bg=self.C_PANEL, fg=self.C_TEXT)

        self.lbl_sample.configure(
            text=f"Sample {self.sample_idx + 1} / {len(self.val_samples)}"
        )
        self.lbl_acc.configure(text="")
        self._anchor_pos = -1

        self.lbl_step.configure(text="Running model...")
        self.root.update()

        self.steps = capture_all_steps(
            self.model_l2r, self.model_r2l,
            sample["prompt_ids"], self.tokenizer, self.device,
        )
        self._target_ids = target_ids

        # Record anchor position (step 1 is where anchor is placed)
        if len(self.steps) > 1:
            self._anchor_pos = self.steps[1]["best_pos"]

        self.current_step = 0
        self._render_step()

    def _new_sample(self):
        self._load_sample(self.sample_idx + 1)

    # ── step navigation ───────────────────────────────────────────────────────

    def _prev_step(self):
        self._stop_play()
        if self.current_step > 0:
            self.current_step -= 1
            self._render_step()

    def _next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._render_step()

    def _toggle_play(self):
        if self._after_id is not None:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.btn_play.configure(text="[Stop]", fg=self.C_WRONG)
        self._auto_advance()

    def _stop_play(self):
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.btn_play.configure(text=">>  Auto Play", fg=self.C_BTN_ACT)

    def _auto_advance(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._render_step()
            self._after_id = self.root.after(self.AUTO_PLAY_MS, self._auto_advance)
        else:
            self._stop_play()

    # ── render ────────────────────────────────────────────────────────────────

    def _tok_str(self, tok_id):
        eos_id = self.tokenizer.eos_token_id
        if tok_id == 0:
            return "[M]"
        if tok_id == eos_id:
            return "<e>"
        s = self.tokenizer.decode([tok_id]).strip()
        return s if s else "."

    def _render_step(self):
        state = self.steps[self.current_step]
        total = len(self.steps) - 1

        if self.current_step == 0:
            info = f"Step 0 / {total}  --  all positions masked"
        else:
            src = state["source"]
            src_label = {"anchor": "ANCHOR", "l2r": "L2R →", "r2l": "← R2L"}.get(src, src.upper())
            info = (
                f"Step {self.current_step} / {total}  |  "
                f"pos {state['best_pos']}  |  "
                f"token '{state['token_str'].strip()}'  |  "
                f"{src_label}  |  "
                f"conf {state['confidence']:.3f}"
            )
        self.lbl_step.configure(text=info)

        eos_id    = self.tokenizer.eos_token_id
        resp      = state["response"]
        filled    = state["filled"]
        l2r_preds = state["l2r_preds"]
        r2l_preds = state["r2l_preds"]
        l2r_confs = state["l2r_confs"]
        r2l_confs = state["r2l_confs"]

        just_pos = state["best_pos"] if self.current_step > 0 else -1
        just_src = state["source"]   if self.current_step > 0 else ""

        for i in range(RESPONSE_LENGTH):
            tok_id    = resp[i]
            is_filled = filled[i]
            just_filled = (i == just_pos)

            # ── response label ──────────────────────────────────────────────
            if not is_filled:
                text = "[MASK]"
                bg   = self.C_MASKED
                fg   = self.C_MUTED
            else:
                text = self.tokenizer.decode([tok_id]) if tok_id != eos_id else "<eos>"
                text = text.strip() or "."
                if just_filled:
                    bg = (self.C_ANCHOR   if just_src == "anchor" else
                          self.C_JUST_L2R if just_src == "l2r"    else
                          self.C_JUST_R2L)
                elif i == self._anchor_pos:
                    bg = self.C_ANCHOR
                else:
                    bg = self.C_FILLED
                fg = "#1e1e2e"
            self.resp_labels[i].configure(text=text, bg=bg, fg=fg)

            # ── L2R / R2L prediction labels ─────────────────────────────────
            lp = l2r_preds[i]
            rp = r2l_preds[i]
            lc = l2r_confs[i]
            rc = r2l_confs[i]
            l2r_wins = lc >= rc

            l2r_txt = f"{self._tok_str(lp)}\n{lc:.2f}"
            r2l_txt = f"{self._tok_str(rp)}\n{rc:.2f}"

            if is_filled:
                self.l2r_labels[i].configure(text=l2r_txt, bg=self.C_PANEL, fg=self.C_MUTED)
                self.r2l_labels[i].configure(text=r2l_txt, bg=self.C_PANEL, fg=self.C_MUTED)
            else:
                l2r_bg = self.C_JUST_L2R if l2r_wins  else self.C_PANEL
                l2r_fg = "#1e1e2e"        if l2r_wins  else self.C_TEXT
                r2l_bg = self.C_JUST_R2L  if not l2r_wins else self.C_PANEL
                r2l_fg = "#1e1e2e"        if not l2r_wins else self.C_TEXT
                self.l2r_labels[i].configure(text=l2r_txt, bg=l2r_bg, fg=l2r_fg)
                self.r2l_labels[i].configure(text=r2l_txt, bg=r2l_bg, fg=r2l_fg)

        # Accuracy — only when fully generated
        if self.current_step == total:
            correct  = all(g == t for g, t in zip(resp, self._target_ids))
            acc_text  = "Correct" if correct else "Incorrect"
            acc_color = self.C_CORRECT if correct else self.C_WRONG
            self.lbl_acc.configure(text=acc_text, fg=acc_color)
            for i, (g, t) in enumerate(zip(resp, self._target_ids)):
                self.tgt_labels[i].configure(fg=self.C_CORRECT if g == t else self.C_WRONG)
        else:
            self.lbl_acc.configure(text="")
            for lbl in self.tgt_labels:
                lbl.configure(fg=self.C_TEXT)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        padding_side="right", use_fast=True,
    )

    print("Loading validation data...")
    val_samples = load_val_samples(tokenizer, args.order, args.num_val)
    if not val_samples:
        raise RuntimeError("No validation samples loaded!")

    model_name = f"Diff_LLaMA_{args.model}M"
    config = Config.from_name(model_name)
    print(f"Loading models ({model_name})...")
    if args.ckpt_path:
        l2r_model = load_model(args.ckpt_path, config, device, state_key="l2r_model")
        r2l_model = load_model(args.ckpt_path, config, device, state_key="r2l_model")
    else:
        l2r_model = load_model(args.l2r_path, config, device, state_key="l2r_model")
        r2l_model = load_model(args.r2l_path, config, device, state_key="r2l_model")

    torch.set_float32_matmul_precision("high")

    print("Launching demo window (requires X11 display)...")
    root = tk.Tk()
    root.geometry("1200x820")
    app = DemoApp(root, val_samples, l2r_model, r2l_model, tokenizer, device)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
