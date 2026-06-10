"""cab-ir-lab: neural guitar cabinet IR simulation (numpy-only smoke implementation)."""

SR = 48_000
N_TAPS = 4096          # 85 ms IR
N_BINS = N_TAPS // 2 + 1  # 2049 rfft bins
