# Feasibility, inventory and download budget

## 1. Inventory: what is actually on this machine

| Requirement | Present |
|---|---|
| `JWST_scnience_env_2` | **No** — no such path anywhere on the filesystem |
| JWST data products (FITS) | **No** |
| `numpy` | **No** |
| `astropy` / `astroquery` | **No** |
| `jwst` pipeline / CRDS | **No** |
| `msaexp` | **No** |
| Reduced spectra, catalogues, prior run outputs | **No** |
| Free disk in the working tree | 16 GB |

The workspace is `rule30-foundry`, a Wolfram Rule 30 repository. Nothing in it
concerns galaxies. **Plan A cannot be executed here**, and no partial execution
would be honest: there is no FITS file to reduce and no library to reduce it
with. This is a missing-environment blocker, not a scientific one.

## 2. The one question that decides everything: the 5.5 µm calibration

Plan A asks whether "the extended NIRSpec calibration reaching approximately
5.5 µm is publicly reproducible". Verified from the source: the extension is
**not a CRDS reference file** — it is "the most recent realization of the data
reduction pipeline provided by the NIRSpec GTO team" (Scholtz et al. 2025),
which "extends the extracted and calibrated spectral traces on the detector"
from 5.3 µm to 5.5 µm.

Consequences, stated as consequences and not as conclusions:

- At z = 9.7995, Hβ (4862.7 Å) falls at **5.2515 µm**, [O III] 4959 at
  **5.3555 µm** and [O III] 5007 at **5.4073 µm** (computed, not recalled:
  λ_obs = λ_rest × (1 + z)). Hβ is inside the nominal PRISM red edge of
  ≈ 5.3 µm; **both [O III] lines are outside it.** The 5007 line is the
  strongest line and the redshift anchor, and it sits wholly in the region the
  extension adds.
- A stock `jwst` pipeline run therefore cannot recover it. The paper's own
  earlier reduction (Fujimoto+24) did not, and reported z = 9.880 instead, which
  the source attributes partly to damping-wing absorption.
- Therefore: unless the GTO realisation is obtained, any reproduction is a
  **partial reproduction** — reproducible to ≈ 5.3 µm (Hβ, the UV lines,
  [O II] at 4.025 µm, [Ne III]3869 at 4.178 µm) and not to [O III] 4959,5007.
- OPEN-4 (is that pipeline public?) is unverified here and must be settled
  first. The honest default is **no**, because team pipeline realisations are
  typically released as method descriptions plus code at best, and the paper
  ships no link.

**Do not** substitute synthetic data for the missing coverage without labelling
it. A synthetic [O III]5007 injected at the expected flux would make a
"successful reproduction" that measures the injection, not the instrument.

## 3. Download budget: no approval transfers to this target

The earlier 6 GiB approval was for a **SMACS** calibration blocker. This target
is Abell 2744 with three separate programmes (2561, 8051, 5578). That approval
is not reused. Current budget: **0 bytes committed, 0 bytes downloaded.**

Sizing procedure — run from the real environment, before downloading anything:

```python
# Estimate before downloading. Prints total bytes, changes nothing.
from astroquery.mast import Observations
for pid in (2561, 8051, 5578):
    obs = Observations.query_criteria(proposal_id=str(pid), obs_collection="JWST")
    prod = Observations.get_product_list(obs)
    for level, mask in (("cal", "CALIBRATED"), ("s2d", "SCIENCE")):
        sel = Observations.filter_products(prod, productType=mask)
        print(pid, level, len(sel), sel["size"].sum() / 1e9, "GB")
```

Then: `sum(size)` gives the real number in seconds, and only the products for
the slitlet/row containing U13151 need fetching. Deliberately not quoted as a
figure here — a guessed size is exactly the kind of number this repository's own
guidance says not to cite (see `AGENTS.md`, "do not cite a number from `data/`
without checking the integrity note" — same principle, applied to a download
estimate).

**Feasible minimal scope**, in order:

1. NIRSpec/PRISM `s2d` for the single MSA slitlet containing U13151 — hundreds
   of MB at most, not the whole MSA exposure set.
2. MIRI/LRS `s2d` + `x1d`, two nod positions — small (MB).
3. NIRCam imaging cutouts for the normalisation — the public UNCOVER DR already
   provides mosaics; cutouts only, not the survey.

That sequence keeps the first useful measurement under ~1 GB and defers the full
NIRCam mosaic set, which is the only part likely to approach gigabyte scale.

Also check proprietary status of **8051** (PRISMS is GTO and may still be in its
exclusive-access window). If it is, Plan A's MIRI leg is blocked for a calendar
reason, not a technical one, and the NIRSpec leg proceeds alone — which is
already the paper's stated order of authority (NIRSpec z is adopted; MIRI is a
cross-check).

## 4. Verdict for Plan A

**BLOCKED**, on two independent and separately falsifiable grounds:

| Blocker | Type | Lifts when |
|---|---|---|
| No JWST environment, no data, no pipeline | Mechanical | Run on `JWST_scnience_env_2` |
| NIRSpec-GTO 5.3→5.5 µm pipeline realisation needed for the 5007 anchor | Scientific/software | OPEN-4 resolved; if unavailable, scope the claim to a **partial reproduction** and say so in the title of the result |

What is *not* blocked: the specification (`02`), the acceptance criteria
(`04`), the internal consistency check (executable here, passing), and the
source verification (`01`). Those are the parts that cost the most to redo and
are now done.
