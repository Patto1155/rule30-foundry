#!/usr/bin/env python
"""Check the frozen U13151 reproduction spec for internal consistency.

Why this exists
---------------
Plan A asks whether a pipeline can reproduce the published emission-line
redshift and fluxes of PRISMS/U13151 (arXiv:2610.26560v1). The first step is to
transcribe that paper's numbers into a frozen specification. Transcribing
numbers is exactly the step where a decimal point moves, and every downstream
fit would then be validated against the wrong target.

So this tool re-derives, from the paper's own table, every quantity the paper
also quotes independently: a redshift uncertainty from a stated velocity
error, a specific star-formation rate from a mass and a rate, a metallicity
percentage from an oxygen abundance. Each is an arithmetic identity, not a
model, so a failure means the specification is wrong - or the paper is.

That distinction matters and is encoded here. A failing check is `FAIL` (the
transcription is wrong; stop). A number the paper states that its own other
numbers contradict is `DISCREPANCY` (record it, do not silently adopt it, and
do not let it become an acceptance criterion). Exit code is 1 only for FAIL,
because a published inconsistency is a finding, not a defect in this tool.

Only the Python standard library is used: this must run before any JWST data,
any astroquery, any numpy, on whatever machine the reproduction is attempted.

Usage:
    python tools/jwst_u13151_consistency.py
    python tools/jwst_u13151_consistency.py --json
    python tools/jwst_u13151_consistency.py --quiet     # exit code only
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

C_KMS = 299792.458

# ---------------------------------------------------------------------------
# Frozen constants, transcribed from arXiv:2610.26560v1 (Rinaldi et al. 2026).
# Every value carries the place in the paper it came from. Do not edit without
# re-reading the source; this file is the reproduction's target.
# ---------------------------------------------------------------------------
SOURCE = "arXiv:2610.26560v1"

# Table 2 (label `tab:spec`), observed fluxes, 1e-20 erg/s/cm^2, NOT corrected
# for lensing magnification. Third element marks an entry as an upper limit.
FLUXES = {
    "N IV]1486": (None, None, True),
    "C IV 1549": (60.9, 20.2, False),
    "O III]1660,1666 + He II 1640": (78.5, 17.2, False),
    "N III]1750": (None, None, True),
    "C III]1907,1909": (41.9, 13.1, False),
    "[O II]3727,3729": (7.7, 3.7, False),
    "[Ne III]3869": (24.1, 3.3, False),
    "He I 3889 + H8": (9.6, 2.6, False),
    "[Ne III]3968 + H epsilon": (15.5, 2.8, False),
    "H delta": (11.4, 5.5, False),
    "H gamma": (21.9, 9.0, False),
    "[O III]4363": (8.6, 4.3, False),
    "H beta": (68.1, 10.5, False),
    "[O III]4959": (78.5, 4.9, False),
    "[O III]5007": (236.3, 14.7, False),
    # MIRI/LRS unresolved complexes; not usable in the optical diagnostics.
    "H beta + [O III] (LRS)": (342.6, 61.1, False),
    "H alpha + [N II] (LRS)": (196.1, 33.3, False),
}

# Redshifts, Section III.2.1. The NIRSpec value is the one the paper adopts.
Z_NIRSPEC, Z_NIRSPEC_ERR = 9.7995, 0.0270
Z_MIRI_HBO3, Z_MIRI_HBO3_LO, Z_MIRI_HBO3_HI = 9.8351, 0.0436, 0.0413
Z_MIRI_HA, Z_MIRI_HA_LO, Z_MIRI_HA_HI = 9.7621, 0.0314, 0.0318
PRIOR_Z_FUJIMOTO = (9.880, 0.001, 0.012)          # Fujimoto et al. 2024
NIRSPEC_PRISM_SIGMA_V_KMS = 750.0                  # Boker et al. 2023
MIRI_LRS_WLCAL_SIGMA_ANGSTROM = 200.0              # "up to ~20 nm", STScI

# Derived quantities quoted in the abstract and in Table 2.
LOG_MSTAR = 6.68                    # abstract, M_sun
SFR_HA, SFR_HA_ERR = 0.45, 0.08     # Table 2, lensing-corrected
SSFR_QUOTED_GYR = 94.0              # abstract
MSTAR_OVER_SFR_QUOTED_MYR = 10.6    # abstract
MOH_TE, MOH_TE_LO, MOH_TE_HI = 7.44, 0.17, 0.33   # Table 2
Z_PERCENT_QUOTED = (3.0, 6.0)       # abstract: "Z ~ 3-6% Zsun"
MU, MU_ERR = 12.8, 0.7              # Furtak et al. 2023; Price et al. 2024

# Asplund et al. 2021, the solar reference the paper cites for abundances.
SOLAR_12_LOGH = 8.69

# Theoretical [O III] 5007/4959 ratio for Case B recombination.
DOUBLET_OIII_THEORY = 2.98

# The paper's own definition of the O32 offset, Table 1 caption.
O32_STATED_OFFSET_DEX = math.log10(1.3)


@dataclass
class Check:
    """One arithmetic identity drawn from the source paper."""

    name: str
    ok: bool
    detail: str
    discrepancy: bool = False
    values: dict = field(default_factory=dict)


def sigma_z_from_velocity(z: float, sigma_v_kms: float) -> float:
    """Redshift uncertainty implied by a velocity uncertainty at redshift z."""
    return (1.0 + z) * sigma_v_kms / C_KMS


def check_wavelength_calibration_sigma() -> Check:
    """The paper footnote: 750 km/s at z~9.8 is 'sigma_z ~ 0.027'."""
    got = sigma_z_from_velocity(Z_NIRSPEC, NIRSPEC_PRISM_SIGMA_V_KMS)
    ok = abs(got - Z_NIRSPEC_ERR) / Z_NIRSPEC_ERR < 0.01
    return Check(
        "sigma_z from 750 km/s wavelength calibration",
        ok,
        f"{got:.4f} vs quoted {Z_NIRSPEC_ERR:.4f} (1% tolerance)",
        values={"derived": got, "quoted": Z_NIRSPEC_ERR},
    )


def check_redshift_consistency() -> list[Check]:
    """MIRI/LRS redshifts must agree with NIRSpec within 1 sigma, as claimed.

    Reported twice on purpose. The paper's '1 sigma' uses MIRI *statistical*
    errors only; the ~20 nm LRS wavelength-calibration systematic, which the
    paper itself flags as a possible cause of the offset, is not in that error
    bar. Both numbers are needed to judge the claim honestly.
    """
    checks = []
    rest = {"H beta + [O III]": 5007.0, "H alpha": 6564.6}
    cases = {
        "H beta + [O III]": (Z_MIRI_HBO3, Z_MIRI_HBO3_LO, Z_MIRI_HBO3_HI),
        "H alpha": (Z_MIRI_HA, Z_MIRI_HA_LO, Z_MIRI_HA_HI),
    }
    for label, (z, lo, hi) in cases.items():
        sigma_stat = 0.5 * (lo + hi)
        diff = z - Z_NIRSPEC
        pull = abs(diff) / math.hypot(sigma_stat, Z_NIRSPEC_ERR)
        sigma_wl = MIRI_LRS_WLCAL_SIGMA_ANGSTROM / rest[label]
        pull_wl = abs(diff) / math.hypot(
            math.hypot(sigma_stat, sigma_wl), Z_NIRSPEC_ERR)
        checks.append(Check(
            f"NIRSpec vs MIRI {label} redshift, statistical errors only",
            pull <= 1.0,
            f"|dz|={abs(diff):.4f}, pull={pull:.2f} sigma "
            f"(paper: consistent within 1 sigma)",
            values={"dz": diff, "pull": pull},
        ))
        checks.append(Check(
            f"NIRSpec vs MIRI {label} redshift, incl. LRS 20 nm wl-cal",
            True,  # informational: no criterion is frozen for this
            f"pull={pull_wl:.2f} sigma once sigma_lambda=20 nm "
            f"(sigma_z={sigma_wl:.4f}) is added in quadrature",
            values={"pull": pull_wl, "sigma_z_wlcal": sigma_wl},
        ))
    return checks


def check_doublet_ratio() -> Check:
    """[O III] 5007/4959 is set by atomic physics (~2.98), not by the fit."""
    f5007, e5007, _ = FLUXES["[O III]5007"]
    f4959, e4959, _ = FLUXES["[O III]4959"]
    ratio = f5007 / f4959
    err = ratio * math.hypot(e5007 / f5007, e4959 / f4959)
    pull = abs(ratio - DOUBLET_OIII_THEORY) / err
    return Check(
        "[O III] 5007/4959 vs theoretical 2.98",
        pull <= 1.0,
        f"{ratio:.3f} +/- {err:.3f}, pull={pull:.2f} sigma; a flux table that "
        "violates this has a transcription or deblending error",
        values={"ratio": ratio, "err": err, "pull": pull},
    )


def check_specific_sfr() -> Check:
    """sSFR(Ha) = SFR(Ha)/M*, as quoted in the abstract (~94 Gyr^-1)."""
    ssfr = SFR_HA / (10.0 ** LOG_MSTAR) * 1e9
    rel = abs(ssfr - SSFR_QUOTED_GYR) / SSFR_QUOTED_GYR
    return Check(
        "sSFR(H alpha) from SFR and M*",
        rel < 0.02,
        f"{ssfr:.1f} Gyr^-1 vs quoted ~{SSFR_QUOTED_GYR:.0f} Gyr^-1",
        values={"derived": ssfr, "quoted": SSFR_QUOTED_GYR},
    )


def check_mstar_over_sfr() -> Check:
    """M*/SFR(Ha) in Myr, the other form of the same identity (~10.6 Myr)."""
    myr = 10.0 ** LOG_MSTAR / SFR_HA / 1e6
    rel = abs(myr - MSTAR_OVER_SFR_QUOTED_MYR) / MSTAR_OVER_SFR_QUOTED_MYR
    return Check(
        "M*/SFR(H alpha) in Myr",
        rel < 0.02,
        f"{myr:.1f} Myr vs quoted {MSTAR_OVER_SFR_QUOTED_MYR} Myr",
        values={"derived": myr, "quoted": MSTAR_OVER_SFR_QUOTED_MYR},
    )


def check_metallicity_percent() -> Check:
    """12+log(O/H)_Te = 7.44 must land inside the abstract's '3-6% Zsun'."""
    pct = 10.0 ** (MOH_TE - SOLAR_12_LOGH) * 100.0
    lo, hi = Z_PERCENT_QUOTED
    return Check(
        "Z_gas from 12+log(O/H)_Te and solar 8.69",
        lo <= pct <= hi,
        f"{pct:.2f}% Zsun vs abstract range {lo:.0f}-{hi:.0f}%",
        values={"percent": pct, "range": [lo, hi]},
    )


def check_o32_definition_offset() -> Check:
    """The O32 convention offset in Table 1 is a rounding, not a discrepancy.

    Table 1 adopts O32 = log10(([O III]4959+5007)/[O II]3727,3729) and states a
    fixed log10(1.3) = 0.114 dex offset from the common
    O32 = log10([O III]5007/[O II]). Re-derived from fluxes the offset is
    log10(1 + F4959/F5007) = 0.125 dex, i.e. their factor is a rounded 1.33.

    The trap this check exists to catch: the same expression built from
    *wavelengths* (1 + 4959/5007 = 1.99 -> 0.299 dex) implies a 0.185 dex
    discrepancy that does not exist. The doublet is a flux ratio of 2.98, not a
    wavelength ratio of 1.01, and anyone re-deriving this from line positions
    alone will manufacture a disagreement. Check it in flux, always.
    """
    f4959 = FLUXES["[O III]4959"][0]
    f5007 = FLUXES["[O III]5007"][0]
    from_fluxes = math.log10(1.0 + f4959 / f5007)
    from_theory = math.log10(1.0 + 1.0 / DOUBLET_OIII_THEORY)
    from_wavelengths = math.log10(1.0 + 4959.0 / 5007.0)
    gap = abs(from_theory - O32_STATED_OFFSET_DEX)
    return Check(
        "O32 convention offset (Table 1) is a rounding of 1.33",
        ok=gap <= 0.02,
        detail=(
            f"stated {O32_STATED_OFFSET_DEX:.3f} dex; from fluxes "
            f"{from_fluxes:.3f}, from the 2.98 doublet {from_theory:.3f} "
            f"(gap {gap:.3f} dex = rounding 1.33 -> 1.3). A wavelength-based "
            f"derivation would give {from_wavelengths:.3f} and a spurious "
            "0.185 dex discrepancy."
        ),
        values={"stated": O32_STATED_OFFSET_DEX, "from_fluxes": from_fluxes,
                "from_theory": from_theory, "from_wavelengths": from_wavelengths},
    )


def derived_diagnostics() -> dict[str, float]:
    """The paper's Table 1 indices, computed from Table 2 fluxes.

    These are the frozen expected values for a reproduction. They are derived
    from the published table only, so they restate the paper's inputs
    arithmetically; they are not an independent measurement of anything.
    """
    hb = FLUXES["H beta"][0]
    hd = FLUXES["H delta"][0]
    hg = FLUXES["H gamma"][0]
    o2 = FLUXES["[O II]3727,3729"][0]
    ne3 = FLUXES["[Ne III]3869"][0]
    o3_4959 = FLUXES["[O III]4959"][0]
    o3_5007 = FLUXES["[O III]5007"][0]
    o3_4363 = FLUXES["[O III]4363"][0]
    return {
        "R3": math.log10(o3_5007 / hb),
        "O32_paper_def": math.log10((o3_4959 + o3_5007) / o2),
        "O32_common_def": math.log10(o3_5007 / o2),
        "R23": math.log10((o3_4959 + o3_5007 + o2) / hb),
        "Ne3O2": math.log10(ne3 / o2),
        "Ne3O2Hd": math.log10((ne3 + o2) / hd),
        "O3Hg": math.log10(o3_4363 / hg),
        "O33": math.log10(o3_5007 / o3_4363),
    }


def documented_refusals() -> list[str]:
    """Ratios a careless pipeline would compute and must not.

    Kept in code so the reason survives into any later session that imports
    this module.
    """
    return [
        "No Balmer decrement from the table's H alpha + [N II] (LRS) over "
        "H beta (NIRSpec): different instruments, apertures (0.44 arcsec for "
        "MIRI/LRS) and normalisations. The paper's own A_V(Ha/Hb) has "
        "sigma = 0.62 mag and is set to 0 throughout.",
        "Upper limits (N IV]1486 < 67, N III]1750 < 46) never enter a ratio; "
        "log10(N/O) < -0.76 is a limit requiring an ICF, not an arithmetic "
        "consequence of the table.",
        "No line flux is lensing-corrected: Table 2 is observed flux, "
        "mu = 12.8 +/- 0.7 is quoted separately, and only the quantities "
        "marked with a dagger are corrected.",
    ]


def summarise(checks: list[Check]) -> tuple[list[Check], list[Check]]:
    """Split checks into (transcription failures, recorded discrepancies).

    A discrepancy is a statement the paper makes that its own numbers
    contradict. It must not fail the build - the build is about *our* work -
    but it must not vanish either.
    """
    fails = [c for c in checks if not c.ok and not c.discrepancy]
    discrepancies = [c for c in checks if c.discrepancy]
    return fails, discrepancies


def run_all() -> tuple[list[Check], dict[str, float]]:
    checks: list[Check] = [
        check_wavelength_calibration_sigma(),
        *check_redshift_consistency(),
        check_doublet_ratio(),
        check_specific_sfr(),
        check_mstar_over_sfr(),
        check_metallicity_percent(),
        check_o32_definition_offset(),
    ]
    return checks, derived_diagnostics()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify the frozen U13151 spec is internally consistent.")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    args = ap.parse_args()

    checks, diag = run_all()
    fails, discrepancies = summarise(checks)

    if args.json:
        print(json.dumps({
            "source": SOURCE,
            "checks": [asdict(c) for c in checks],
            "derived_diagnostics": diag,
            "documented_refusals": documented_refusals(),
            "n_fail": len(fails),
            "n_discrepancy": len(discrepancies),
        }, indent=2))
        return 1 if fails else 0

    if not args.quiet:
        print(f"jwst_u13151_consistency: spec frozen from {SOURCE}")
        print()
        for c in checks:
            tag = "DISCREPANCY" if c.discrepancy else ("PASS" if c.ok else "FAIL")
            print(f"  {tag:11s} {c.name}")
            print(f"              {c.detail}")
        print()
        print("  Derived diagnostics (paper Table 1 indices, from Table 2 "
              "fluxes):")
        for key, value in diag.items():
            print(f"    {key:16s} {value:+.4f}")
        print()
        print("  Documented refusals (ratios a careless pipeline would "
              "compute):")
        for note in documented_refusals():
            print(f"    - {note}")
        print()
        if fails:
            print(f"  {len(fails)} FAIL: the transcription is wrong. Stop.")
        elif discrepancies:
            print(f"  {len(discrepancies)} DISCREPANCY: the paper contradicts "
                  "itself; record it, do not adopt either branch silently.")
        else:
            print("  All consistency checks pass.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
