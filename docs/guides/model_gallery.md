# Model Gallery (Notebook)

This gallery notebook walks through the core simulators—Ball, Stick, Zeppelin, DTI, Sphere, Cylinder, NODDI (Watson/Bingham), and SANDI (Watson/Bingham). Use it as a quick reference when exploring models or building new mixtures.

- Notebook: `examples/model_gallery.ipynb`
- Run locally to interact; key signals are previewed below.

## Quick setup

```python
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
from dmri.simulators import Ball, Stick, Zeppelin, Dti, Sphere, Cylinder, NoddiW, NoddiB, SandiW, SandiB
from dmri.simulators.acquisition_scheme import acquisition_scheme

bvals = jnp.linspace(0, 3000, 60)
bvecs = jax.random.normal(jax.random.key(0), (60, 3))
bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
acq = acquisition_scheme(bvals, bvecs)
```

## Preview plots

![Ball](../assets/gallery_ball.svg)

![Stick](../assets/gallery_stick.svg)

![Zeppelin](../assets/gallery_zeppelin.svg)

![DTI](../assets/gallery_dti.svg)

For restricted models (Sphere, Cylinder) and NODDI/SANDI variants, run the notebook to render their signals interactively.

## Next steps

- Vary `mu`, `lambda_par`, `lambda_perp`, and `radius` to see orientation and restriction effects.
- Swap in noise compartments when simulating magnitude data.
- Combine these models with `MultiCompartment` to form richer mixtures (e.g., Ball+Stick+Zeppelin).
