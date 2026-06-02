OUGS — Dual-License Notice
==========================

This repository is dual-licensed.

1. **Upstream code** inherited from 3D Gaussian Splatting (Kerbl et al., 2023) is
   distributed under the original Inria / MPII non-commercial research licence —
   see [`LICENSE.md`](LICENSE.md). That licence applies to the files this project
   inherited and modified from the upstream codebase, including (non-exhaustive):

   ```
   gaussian_renderer/    scene/    utils/    lpipsPyTorch/    arguments/
   submodules/    train.py    render.py    metrics.py    convert.py
   extract_depth.py
   ```

2. **OUGS original contributions** — files and additions authored for the OUGS
   project — are licensed under the **Creative Commons Attribution 4.0
   International License (CC-BY 4.0)**:

   https://creativecommons.org/licenses/by/4.0/legalcode

   This matches the open-access licence of the OUGS paper published in
   *Computer Graphics Forum* (Eurographics 2026, DOI
   [10.1111/cgf.70363](https://doi.org/10.1111/cgf.70363)).

   The CC-BY 4.0 licence applies to the following directories and to any clearly
   OUGS-original additions inside the upstream files above (those additions can
   be identified from the git history of this repository):

   ```
   pipeline/        evaluation/      preprocess/      scripts/
   docs/            assets/          README.md
   ```

   You are free to **share** (copy and redistribute) and **adapt** (remix,
   transform, build upon) the OUGS-original material for any purpose, including
   commercial use, provided that you give appropriate credit, provide a link to
   the licence, and indicate if changes were made.

----

Suggested attribution
---------------------

> Li, H., Chen, Q., Kalkofen, D., Chen, H.-T. *OUGS: Active View Selection via
> Object-aware Uncertainty Estimation in 3DGS.* Computer Graphics Forum 45,
> Eurographics 2026. DOI: 10.1111/cgf.70363.

```bibtex
@article{10.1111:cgf.70363,
  journal   = {Computer Graphics Forum},
  title     = {{OUGS: Active View Selection via Object-aware Uncertainty Estimation in 3DGS}},
  author    = {Li, Haiyi and Chen, Qi and Kalkofen, Denis and Chen, Hsiang-Ting},
  year      = {2026},
  publisher = {The Eurographics Association and John Wiley \& Sons Ltd.},
  ISSN      = {1467-8659},
  DOI       = {10.1111/cgf.70363}
}
```

----

Copyright (c) 2026 Haiyi Li, Qi Chen, Denis Kalkofen, Hsiang-Ting Chen
(The University of Adelaide · Graz University of Technology).

The OUGS-original contributions in this repository are released under
CC-BY 4.0. The upstream 3D Gaussian Splatting code retains its original
non-commercial research licence as detailed in `LICENSE.md`.
