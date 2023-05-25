# 1st batch: baselines
python tests/benchmark/profiling.py -f                  # baseline (fastest config possible)
python tests/benchmark/profiling.py -s Rs_int -f        # for vision research
python tests/benchmark/profiling.py -s Rs_int -r        # for robotics research

# 2nd batch: compare different scenes
python tests/benchmark/profiling.py -f -r
python tests/benchmark/profiling.py -f -r -s Rs_int
python tests/benchmark/profiling.py -f -r -s Benevolence_0_int
python tests/benchmark/profiling.py -f -r -s Wainscott_0_int

# 3rd batch: OG non-physics features
python tests/benchmark/profiling.py -r -s Rs_int -w             # fluids (water)
python tests/benchmark/profiling.py -r -s Rs_int -c             # soft body (cloth)
python tests/benchmark/profiling.py -r -s Rs_int -p             # macro particle system (diced objects)
python tests/benchmark/profiling.py -r -s Rs_int -w -c -p       # everything (slowest config possible)