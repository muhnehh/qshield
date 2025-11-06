# qshield
## 🚀 QShield — Hybrid Quantum-Safe Secure Messenger

![status](https://img.shields.io/badge/status-planning-blue)
![python](https://img.shields.io/badge/python-3.11+-informational)
![security](https://img.shields.io/badge/security-QKD%20%2B%20PQC-critical)

Welcome — this README has been rewritten for clarity, friendliness, and a little flair ✨. The goal: make the project instantly approachable for contributors and memorable for readers.

If you're looking for the original README, it's archived at the bottom under "Original README (archived)".

---

## 📌 Quick elevator pitch

- QShield is an experimental secure chat that combines quantum key distribution (BB84 simulated with Cirq) and post-quantum KEMs (Kyber via OQS) to establish session keys. The data plane uses AES-GCM. Think of it as a research-grade, hybrid approach to “quantum-safe” messaging.

Why this README feels different: clearer structure, helpful badges, PowerShell-friendly commands, emoji signposts, and an easy “Try it” quick start.

---

## ✨ Highlights (what makes QShield cool)

- 🔐 Hybrid keying: QKD (BB84) for continuous secret-rate experiments and PQC (Kyber) for robust fallback.
- ⚖️ Measurables: QBER, secret-key rate, reconciliation leakage, and rekey latency.
- ⚙️ Modular: clear `src/` layout (crypto, qkd, net, metrics, policy).
- 🧪 Reproducible: seeded notebooks and deterministic unit tests for the research bits.

---

## 📚 Table of contents

1. [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
2. [Usage examples](#usage-examples)
3. [Architecture at a glance](#architecture-at-a-glance)
4. [Roadmap (short)](#roadmap-short)
5. [Contributing & quality gates](#contributing--quality-gates)
6. [Appendix: original README (archived)](#appendix-original-readme-archived)

---

## Quick start (Windows PowerShell)

These commands are tailored for Windows PowerShell (your default shell). They set up a virtual environment, install dev extras, and run tests.

```powershell
# 1) Clone & create venv
git clone https://github.com/yourname/qshield.git
cd qshield
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Install dependencies (editable), adjust extras in pyproject.toml
python -m pip install --upgrade pip wheel
python -m pip install -e ".[dev]"

# 3) Pre-commit hooks & tests
pre-commit install
pytest -q
```

If you prefer WSL or bash, replace the activation line with `source .venv/bin/activate`.

---

## Usage examples

Start a simple server and connect with a client (auto-mode: QKD preferred, PQC fallback):

```powershell
# run server
python -m qshield.net.server --host 127.0.0.1 --port 8765

# connect client (auto negotiation)
python -m qshield.net.client --url ws://127.0.0.1:8765 --mode auto
```

Pro tip: run `python -m qshield.net.client --help` for CLI flags (mode selection, verbosity, metrics logging).

---

## Architecture at a glance

High-level flow (ASCII diagram):

```
  [ Client ] <--WebSocket--> [ Server ]
      |                         |
      |--- Handshake (QKD or PQC, authenticated) ---|
      |                         |
  [Key Derivation] -> [AES-GCM data plane] -> [Rekey Policy]
      |
  [Key Buffer] <= QKD pipeline: BB84 -> Sift -> Reconcile -> PrivacyAmp
```

Key modes:
- QKD path: simulated BB84 (Cirq) -> sifting -> error reconciliation -> privacy amplification -> key buffer
- PQC path: Kyber KEM (OQS) for quick, robust provisioning

---

## Roadmap (short & actionable)

- Phase 0: repository scaffolding, CI, pre-commit, basics ✅
- Phase 1: AEAD baseline (AES-GCM + HKDF + tests)
- Phase 2: BB84 simulation + QBER metrics
- Phase 3: Sifting, EC, Privacy Amplification (leakage accounting)
- Phase 4: PQC KEM mode (Kyber) + benchmarks
- Phase 5: Authenticated transcripts (Dilithium/HMAC)
- Phase 6: Policy selection (when to use QKD vs PQC)

Stretch: E91, group chat, GUI, quantum experiments

---

## Contributing & quality gates

- Please open small PRs. Each PR should include at least one test.
- CI checks: lint (black/isort), typecheck (mypy), tests (pytest).
- Security checklist for PRs changing crypto:
  - Unique nonce per key
  - Transcript authentication
  - Leakage accounting for reconciliation

If you'd like to help, look for issues labeled `good-first-issue` and `area/qkd`.

---

## Visual & memorable extras

- ASCII logo (feel free to replace with an SVG in `assets/`):

```
  ____  _     _ _ _ _     _ _
 / __ \| |   (_) | (_)   | | |
| |  | | |__  _| | |_  __| | | ___ _ __
| |  | | '_ \| | | | |/ _` | |/ _ \ '__|
| |__| | | | | | | | | (_| | |  __/ |
 \____/|_| |_|_|_|_|_|\__,_|_|\___|_|

     Q S H I E L D  —  Quantum + PQC hybrid
```

- Badges: use shields.io to create additional badges (CI, coverage, docs).

---

## Examples of useful commands

- Run the QKD rate benchmark (small):

```powershell
python benchmarks/qkd_rate.py --p_flip 0.02 --blocks 128
```

- Run KEM latency micro-benchmark:

```powershell
python benchmarks/kem_latency.py --trials 200
```

---

## Files & where to look first

- `src/qshield/crypto/` — AEAD, KEM, signatures
- `src/qshield/qkd/` — BB84, sifting, reconciliation, privacy amplification
- `src/qshield/net/` — server & client (FastAPI + WebSockets)
- `docs/` — protocol, threat model, ADRs
- `notebooks/` — reproducible experiments and plots

---

## License

Choose MIT or Apache-2.0 (pick one in `LICENSE`). Note: third-party crypto components such as OQS/Cirq have their own licensing—please review and attribute accordingly.

---

## Appendix — Original README (archived)

The original README content is preserved below for reference and reproducibility. If you want it restored as the main README, tell me and I can revert or move the new content to `README.enhanced.md` instead.

<!-- original content follows -->

```
# QShield — Hybrid Quantum-Safe Secure Messenger (QKD + PQC)

> End-to-end encrypted chat where session keys are negotiated via **simulated BB84 QKD (Cirq)** or **post-quantum KEM (Kyber via Open Quantum Safe)**, with authenticated transcripts, error correction, privacy amplification, and metrics.

[![Status](https://img.shields.io/badge/status-planning-blue)]() [![Python](https://img.shields.io/badge/python-3.11+-informational)]() [![Security](https://img.shields.io/badge/security-QKD%20%2B%20PQC-critical)]()

---

## 0) TL;DR

- Data plane: **AES-GCM**
- Keying modes: **BB84 (Cirq)** or **Kyber KEM (OQS)** with **Dilithium** signatures or **HMAC** for auth
- Pipeline (QKD): **Sifting → Error Reconciliation → Privacy Amplification → Key Buffer**
- Transport: **FastAPI/WebSockets** (CLI first, GUI later)
- Goals: measurable **secret-key rate**, **QBER**, and end-to-end latency with reproducible experiments

---

## 1) Repository Structure

```
qshield/
├─ README.md
├─ pyproject.toml           # poetry/pip-tools; lock your deps
├─ .pre-commit-config.yaml  # black, isort, flake8, mypy
├─ src/qshield/
│  ├─ __init__.py
│  ├─ common/
│  │  ├─ config.py
│  │  ├─ logging.py
│  │  └─ messages.py
│  ├─ crypto/
│  │  ├─ aead.py            # AES-GCM, HKDF
│  │  ├─ kem.py             # Kyber KEM (python-oqs) + X25519 fallback
│  │  └─ sig.py             # Dilithium / HMAC auth
│  ├─ qkd/
│  │  ├─ bb84.py            # Cirq circuits + noise models
│  │  ├─ sifting.py
│  │  ├─ reconcile.py       # parity blocks → Cascade-lite
│  │  ├─ privacy_amp.py     # Toeplitz hashing
│  │  └─ utils.py
│  ├─ net/
│  │  ├─ server.py          # FastAPI + WebSockets
│  │  └─ client.py          # CLI client
│  ├─ policy/
│  │  └─ selector.py        # choose QKD vs PQC, rekey policies
│  └─ metrics/
│     ├─ monitor.py         # live metrics + CSV logs
│     └─ plots.py
├─ tests/
│  ├─ test_aead.py
│  ├─ test_bb84.py
│  ├─ test_reconcile.py
│  └─ test_kem.py
├─ benchmarks/
│  ├─ qkd_rate.py
│  └─ kem_latency.py
├─ docs/
│  ├─ protocol.md           # message formats, state machines
│  ├─ threat-model.md       # STRIDE + QKD-specific attacks
│  ├─ security.md           # crypto invariants, key lifecycles
│  ├─ roadmap.md
│  └─ adr/                  # Architecture Decision Records
└─ notebooks/
   ├─ qber_vs_noise.ipynb
   └─ pa_compression.ipynb
```

---

## 2) Architecture (at a glance)

```
[ Client ] <--WebSocket--> [ Server ]
     |                          |
     |--- Handshake (QKD or PQC, authenticated) ---|
     |                          |
[Key Derivation] → [AES-GCM Data Plane] → [Rekey Policy]
     |
[Key Buffer] <= QKD pipeline: BB84 → Sift → Reconcile → Privacy Amp
```

---

## 3) Long-Term Roadmap (Work in Small, Independent Chunks)

Each phase is self-contained; open a milestone per phase and convert checkboxes into GitHub issues.

### Phase 0 — Project Scaffolding
- [ ] `pyproject.toml`, `src/` layout, `tests/`, `pre-commit`
- [ ] CI: lint + typecheck + unit tests
- [ ] `docs/adr/0001-toolchain.md` (poetry/pip-tools, mypy, pytest)

**Deliverables:** buildable env, first CI pass

---

### Phase 1 — Classical AEAD Baseline
- [ ] `crypto/aead.py`: AES-GCM (random IV, AAD), HKDF
- [ ] `tests/test_aead.py`: deterministic vectors
- [ ] Minimal encrypt/decrypt CLI

**Deliverables:** baseline secure channel with static PSK

---

### Phase 2 — BB84 Simulation (Cirq)
- [ ] `qkd/bb84.py`: bases, prep/measure, noise (depol, bit-flip)
- [ ] `tests/test_bb84.py`: QBER rises with noise; seeded reproducibility
- [ ] `notebooks/qber_vs_noise.ipynb`

**Deliverables:** BB84 raw key + QBER metrics

---

### Phase 3 — Sifting, Error Reconciliation, Privacy Amplification
- [ ] `sifting.py`: mask + sifted keys
- [ ] `reconcile.py`: parity blocks → **Cascade-lite** (iterative)
- [ ] `privacy_amp.py`: Toeplitz hashing (SHA-256 PRG)
- [ ] Leakage accounting (bits revealed during EC)
- [ ] `tests/test_reconcile.py`, finite-key sanity checks

**Deliverables:** identical reconciled keys; PA output tuned vs QBER

---

### Phase 4 — PQC KEM Mode
- [ ] `crypto/kem.py`: Kyber (OQS); X25519 fallback if OQS missing
- [ ] Bench `benchmarks/kem_latency.py`
- [ ] Compare “time-to-ready key” QKD vs PQC

**Deliverables:** working PQC path with metadata transcript

---

### Phase 5 — Authentication of Transcripts
- [ ] `crypto/sig.py`: **Dilithium** (OQS) + HMAC(PSK) option
- [ ] Auth classical QKD messages + KEM transcript
- [ ] MITM tests (tamper → reject)

**Deliverables:** authenticated handshakes; negative tests

---

### Phase 6 — Policy & Handshake Negotiation
- [ ] `policy/selector.py`: choose QKD if QBER<τ & key_buffer>Kmin; else PQC
- [ ] State machine in `docs/protocol.md`
- [ ] Unit tests of transitions & fallbacks

**Deliverables:** robust mode selection

---

### Phase 7 — Networking Layer
- [ ] `net/server.py` (FastAPI/WebSockets) + `net/client.py` (CLI)
- [ ] Encrypted chat frames (AES-GCM) with rolling nonces
- [ ] Rekey every N messages / T seconds

**Deliverables:** end-to-end encrypted chat (CLI demo)

---

### Phase 8 — Key Buffer & Lifecycle
- [ ] Background QKD producer → bounded buffer
- [ ] Exhaustion handling (pause/send via PQC)
- [ ] Secure in-mem storage + optional SQLite envelope encryption

**Deliverables:** stable long-running sessions

---

### Phase 9 — Metrics & Visualization
- [ ] `metrics/monitor.py`: CSV logs (QBER, leakage, PA rate, key buffer)
- [ ] `metrics/plots.py`: publishable figures
- [ ] `benchmarks/qkd_rate.py`: secret-key rate vs noise/block size

**Deliverables:** reproducible plots + CSVs

---

### Phase 10 — Security Docs
- [ ] `docs/threat-model.md`: eavesdropping, detector noise, MITM, replay
- [ ] `docs/security.md`: invariants, nonce rules, KDF chains, PA math
- [ ] ADRs for major crypto choices

**Deliverables:** review-ready security package

---

### Phase 11 — Stretch Goals
- [ ] E91 variant / decoy states
- [ ] TLS proxy mode (“quantum-safe VPN”)
- [ ] Group chat (TreeKEM + QKD feed)
- [ ] Simple GUI (Textual/Qt)

---

## 4) Getting Started (Dev)

```bash
# 1) Clone & create env
git clone https://github.com/yourname/qshield.git
cd qshield
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -U pip wheel
pip install -e ".[dev]"  # define extras in pyproject (oqs, cirq, fastapi, uvicorn, numpy, pandas, matplotlib, mypy, pytest)

# 3) Pre-commit + tests
pre-commit install
pytest -q
```

---

## 5) Minimal Usage (once phases 1–4 are in)

```bash
# Start server
python -m qshield.net.server --host 127.0.0.1 --port 8765

# Client (QKD preferred, fallback to PQC)
python -m qshield.net.client --url ws://127.0.0.1:8765 --mode auto
```

---

## 6) Testing & Benchmarks

- Unit tests: `pytest -q`
- Type checks: `mypy src/`
- QKD rate: `python benchmarks/qkd_rate.py --p_flip 0.02 --blocks 128`
- KEM latency: `python benchmarks/kem_latency.py --trials 200`

---

## 7) Definition of Done (per phase)

- ✅ Unit tests ≥ 90% for the module
- ✅ Negative tests (tamper, replay, wrong tag)
- ✅ Docs updated (`protocol.md`, ADR if architecture changed)
- ✅ Benchmarks (if perf-relevant)
- ✅ Reproducible seed used in notebooks

---

## 8) Issue Labels (work in slices)

- `area/qkd`, `area/pqc`, `area/net`, `area/crypto`, `area/metrics`
- `type/bug`, `type/feat`, `type/docs`, `type/refactor`
- `prio/P0..P3`
- `good-first-issue` (small, testable tasks)

---

## 9) Research & Notes

Track open questions in `docs/roadmap.md`:
- Finite-key corrections for PA length vs QBER
- Cascade parameter tuning vs block size/noise
- Authentication choices: HMAC(PSK) vs Dilithium (trade-offs)

---

## 10) Security Invariants (quick checklist)

- [ ] Unique nonce per AES-GCM key
- [ ] All transcripts authenticated (HMAC or Dilithium)
- [ ] EC leakage accounted in PA length
- [ ] Rekey before nonce space exhaustion
- [ ] Zeroize ephemeral keys on error/close

---

## 11) License & Attribution

- Code: MIT/Apache-2.0 (choose)
- PQC primitives via **Open Quantum Safe (OQS)**
- Quantum circuits via **Cirq**

---

### Appendix A — Starter Signatures

```python
# src/qshield/qkd/bb84.py
def bb84_run(n_bits: int, p_flip: float = 0.02, seed: int | None = 0) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (alice_bits, alice_bases, bob_bits, bob_bases)."""

# src/qshield/qkd/reconcile.py
def cascade_lite(a_key: bytes, b_key: bytes, block_size: int = 128, rounds: int = 3, seed: int | None = 0) -> tuple[bytes, bytes, int]:
    """Return (a_rec, b_rec, leakage_bits)."""

# src/qshield/qkd/privacy_amp.py
def toeplitz_hash(key_material: bytes, out_len: int, seed: bytes) -> bytes:
    """Universal hashing for PA."""
```

```
