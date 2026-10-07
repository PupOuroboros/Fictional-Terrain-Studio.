from __future__ import annotations

from typing import Any, Iterable


def effective_grade_cutoff_percent(cfg: Any) -> float:
    """Return the active ordinary-development hard cutoff.

    A Bible profile may tighten the legacy target_grade_percent but never
    relax it. This makes the design profile operational in QA/UI/agent views.
    """
    target = float(getattr(cfg, "target_grade_percent", 10.0))
    profile = getattr(cfg, "bible_profile", None)
    if profile is None or not bool(getattr(profile, "enabled", False)):
        return target
    return min(target, float(profile.ordinary_development_hard_cutoff_percent))


def validate_bible_profile(cfg: Any, objects: Iterable[Any] | None = None) -> dict[str, Any]:
    """Validate machine-checkable Bible rules and expose declared doctrine."""
    p = getattr(cfg, "bible_profile", None)
    if p is None or not bool(getattr(p, "enabled", False)):
        return {"enabled": False, "status": "DISABLED", "checks": {}, "warnings": []}

    eps = 1e-6
    vanilla = str(p.map_profile).lower() == "vanilla"
    checks = {
        "canonical_playable_extent": (not vanilla) or abs(float(cfg.extent_km) - float(p.canonical_playable_extent_km)) <= eps,
        "canonical_world_extent": (not vanilla) or abs(float(cfg.production.nominal_world_extent_km) - float(p.canonical_world_extent_km)) <= eps,
        "world_is_four_times_playable": abs(float(p.canonical_world_extent_km) / max(float(p.canonical_playable_extent_km), eps) - 4.0) <= eps,
        "world_center_fraction_quarter": abs(float(cfg.production.world_center_fraction) - 0.25) <= eps,
        "ordinary_development_cutoff_not_relaxed": float(cfg.target_grade_percent) <= float(p.ordinary_development_hard_cutoff_percent) + eps,
        "hydrology_enabled_when_locked": (not p.hydrography_locked) or bool(cfg.hydrology.enabled),
        "post_water_hydrology_qa_available": (not p.post_water_hydrology_qa_required) or bool(cfg.hydrology.enabled),
        "gis_crs_declared": (not bool(getattr(getattr(cfg, "gis", None), "enabled", False))) or bool(str(getattr(getattr(cfg, "gis", None), "crs_wkt", "")).strip()),
    }
    warnings: list[str] = []
    if vanilla and not checks["canonical_playable_extent"]:
        warnings.append(
            f"Legacy/custom playable extent {float(cfg.extent_km):.3f} km differs from the canonical vanilla {float(p.canonical_playable_extent_km):.3f} km profile."
        )
    if not checks["ordinary_development_cutoff_not_relaxed"]:
        warnings.append(
            f"target_grade_percent={float(cfg.target_grade_percent):.2f}% relaxes the Bible hard cutoff of {float(p.ordinary_development_hard_cutoff_percent):.2f}%."
        )

    water_types = {"lake", "coast", "sea", "shoreline", "harbor", "river", "water_source"}
    water_object_count = 0
    for obj in list(objects or []):
        if not getattr(obj, "enabled", True):
            continue
        typ = str(getattr(obj, "object_type", "")).lower().replace("-", "_")
        if typ in water_types:
            water_object_count += 1

    return {
        "enabled": True,
        "profile_name": p.profile_name,
        "profile_version": p.profile_version,
        "map_profile": p.map_profile,
        "status": "PASS" if all(checks.values()) else "WARN",
        "checks": checks,
        "warnings": warnings,
        "effective_ordinary_development_cutoff_percent": effective_grade_cutoff_percent(cfg),
        "buildability_bands_percent": {
            "normal_max": float(p.normal_grade_max_percent),
            "caution_max": float(p.caution_grade_max_percent),
            "constrained_max": float(p.constrained_grade_max_percent),
            "hard_cutoff": float(p.ordinary_development_hard_cutoff_percent),
        },
        "terrain_water_doctrine": {
            "hydrography_locked": bool(p.hydrography_locked),
            "preserve_macro_landforms": bool(p.preserve_macro_landforms),
            "canonical_water_level_required": bool(p.canonical_water_level_required),
            "river_first_workflow": bool(p.river_first_workflow),
            "natural_bathymetry_before_dredging": bool(p.natural_bathymetry_before_dredging),
            "post_water_hydrology_qa_required": bool(p.post_water_hydrology_qa_required),
            "harbor_engineering_separate": bool(p.harbor_engineering_separate),
            "harbor_navigation_transition_required": bool(p.harbor_navigation_transition_required),
        },
        "semantic_water_object_count": int(water_object_count),
        "canonical_extents_km": {
            "playable": float(p.canonical_playable_extent_km),
            "world": float(p.canonical_world_extent_km),
            "ratio": float(p.canonical_world_extent_km) / max(float(p.canonical_playable_extent_km), eps),
        },
    }
