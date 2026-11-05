#!/usr/bin/env python
"""cardiac_ru_driver.py -- run gb_real_uniform's `cpu` / `gpu` / `smoke` for CARDIAC WITHOUT editing gb_real_uniform.py (its SYSTEMS tuple hard-codes the 7 frozen systems and is also the
default set of `eval`, whose hash-pinned figure contract must not gain a system).  Only the module-global tuple is extended IN MEMORY for this process; no file is modified.
  python scratch/cardiac_ru_driver.py cpu CARDIAC [--workers 32]      (Pearson roster + roster-only GENIE3/GRNBoost2 + information-matched; needs the GB cop_alignment cache first)
  python scratch/cardiac_ru_driver.py gpu CARDIAC --gpu 4 --method velorama|regvelo
NOT run by the builder (task scope: no scoring)."""
import sys
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_real_uniform as RU
RU.SYSTEMS = RU.SYSTEMS + ("CARDIAC",)
a = RU.parser().parse_args(sys.argv[1:])
{"cpu": RU.run_cpu, "gpu": RU.run_gpu, "smoke": RU.run_smoke}[a.cmd](a)
