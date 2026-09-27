# Numerical reference and scope

- Reference: [abmasud1214/pufferdle](https://github.com/abmasud1214/pufferdle).
- Pinned commit: `506322f56b2ae1975e0c896fe7bccb731e698fac`.
- Dynamics: [`src/Components/FishingGame.js`](https://github.com/abmasud1214/pufferdle/blob/506322f56b2ae1975e0c896fe7bccb731e698fac/src/Components/FishingGame.js).
- Catalog: [`src/fishdata.js`](https://github.com/abmasud1214/pufferdle/blob/506322f56b2ae1975e0c896fe7bccb731e698fac/src/fishdata.js). Only factual name, difficulty and movement category fields are retained in `fishing_sim/fish.json` (55 entries). Descriptions, artwork, seasons, locations and UI code are not included.
- Algorithm/API: [PPO paper](https://arxiv.org/abs/1707.06347), [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Gymnasium environment API](https://gymnasium.farama.org/api/env/).
- Exact downloaded-file hashes are in `source_manifest.json`. Reference downloads are local and ignored by Git. `fetch_reference.ps1` retrieves the pinned version for comparison.

## What this implementation models

The Python numerical model reproduces the reference fish target changes, five movement categories, fish-speed smoothing, green-bar gravity and inertia, collision rebound, hit test, capture progress, fishing-level bar height and none/Cork/Lead/Trap/Barbed tackle effects. `reference_check.mjs` extracts only the source's fish/bar numerical update and compares the Python implementation under identical random draws. The source's own initialization values are retained.

Coordinate conventions matter: the reference bar occupies y=6..288. Its fish catch hitbox is [fishPos+8, fishPos+22]; our measurement reports its midpoint `fishPos+15`. This is a numerical hitbox convention, not a claim about a detected sprite's visual center. The future image adapter must calibrate this offset.

## Intentional differences

1. Fixed 60 Hz physics and 30 Hz decisions, independent of wall-clock speed. Training runs as fast as the CPU allows.
2. Seeded fish randomness has its own stream. Browser shake/cosmetic random draws are omitted, so the same source seed is not expected to replay the same browser trajectory. Distributional movement rules and controlled-random-input transitions are compared instead.
3. Catch/loss terminates immediately after the decisive tick. Upstream stops its loop on the next update and has callback/animation delays.
4. Treasure is omitted for this catch-first stage. Treasure Hunter, automatic hooking, casting, Sonar, bait, rod enchantments, advanced rods, multi-tackle effects and version-specific modifiers are not implemented.
5. A visible 120-second deadline is added as a task failure to prevent unbounded episodes.
6. Rendering uses original simple shapes. No game sprites or art are required.
7. Reference `floaterSinkerAcceleration` is reinitialized inside each update. The ±0.01 per-tick bias is deliberately preserved; this is not silently changed to accumulated acceleration.

There is no claim of equivalence to every Stardew Valley build. The catalog lacks later additions such as Goby. Reference rights are not a blanket redistribution license: the Pufferdle README credits the original minigame code and assets to ConcernedApe, and no explicit project license was visible. This public repository credits those sources without asserting a project-wide license over their code or artwork. Source visibility and numerical parity do not settle rights questions.

## Live audio fingerprint

`auto_assets/bite_signature.npz` contains a 153-by-10 spectral feature matrix derived from a user-provided gameplay recording (`2026-09-28 01-15-38.mkv`, approximately 146.005–146.144 s). It contains no waveform, speech, music track, or video. The original game sound belongs to its respective rights holder. Parameters and the artifact hash are in `auto_assets/bite_signature.json`; the numerical extractor is `training/build_bite_signature.py`. The audio detector uses this fixed sound-effect signature and an adaptive background estimate, not another trained neural network.
