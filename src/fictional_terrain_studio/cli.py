from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .analysis import terrain_report
from .design_profile import effective_grade_cutoff_percent, validate_bible_profile
from .config import DonorLayerConfig, TerrainConfig
from .export import export_composer_outputs, export_geomorphology_outputs, export_hydrology_outputs, export_outputs, export_spline_outputs
from .generator import generate_terrain
from .composer import load_donor
from .landforms import detect_landform_candidates, extract_landform, save_landform_asset
from .library import load_landform_index, search_landforms, write_landform_index
from .hydrology import analyze_hydrology
from .project import TerrainProject
from .cs2_production import export_cs2_production


def cmd_generate(recipe: str, out_dir: str) -> int:
    cfg = TerrainConfig.from_yaml(recipe)
    elev, meta = generate_terrain(cfg)
    hydrology = None
    if cfg.hydrology.enabled:
        hydrology = analyze_hydrology(
            elev,
            meta["water_mask"],
            cfg.extent_km,
            fill_epsilon_m=cfg.hydrology.fill_epsilon_m,
            stream_threshold_km2=cfg.hydrology.stream_threshold_km2,
            floodplain_hand_m=cfg.hydrology.floodplain_hand_m,
            floodplain_slope_percent=cfg.hydrology.floodplain_slope_percent,
        )

    exported = export_outputs(elev, meta["water_mask"], cfg, out_dir, write_heightmap=not cfg.production.enabled)
    production = None
    if cfg.production.enabled:
        production = export_cs2_production(elev, meta["water_mask"], cfg, out_dir)
    if cfg.composer.enabled:
        export_composer_outputs(meta, cfg, out_dir)
    if cfg.geomorphology.enabled:
        export_geomorphology_outputs(meta, cfg, out_dir)
    if cfg.splines.enabled:
        export_spline_outputs(meta, cfg, out_dir)
    if hydrology is not None:
        export_hydrology_outputs(hydrology, cfg, out_dir)

    report = terrain_report(exported.get("diagnostic_elevation", exported["elevation_export"]), cfg.extent_km, effective_grade_cutoff_percent(cfg))
    report.update(
        {
            "name": cfg.name,
            "seed": cfg.seed,
            "export_resolution": cfg.export_resolution,
            "export_base_level_m": cfg.export_base_level_m,
            "export_elevation_scale_m": cfg.export_elevation_scale_m,
            "river_enabled": cfg.river.enabled,
            "hydrology_enabled": cfg.hydrology.enabled,
            "geomorphology_enabled": cfg.geomorphology.enabled,
            "composer_enabled": cfg.composer.enabled,
            "semantic_splines_enabled": cfg.splines.enabled,
            "cs2_production_enabled": cfg.production.enabled,
            "semantic_spline_count": len(meta.get("semantic_splines", [])),
            "composer_donor_count": len(meta.get("composer_layers", [])),
            "composer_landform_count": len(meta.get("composer_landforms", [])),
            "composer_max_abs_delta_m": float(abs(meta.get("composer_delta")).max()) if cfg.composer.enabled else 0.0,
            "authored_tributaries": len(meta.get("tributary_paths", [])),
        }
    )
    if cfg.composer.enabled:
        delta = np.asarray(meta.get("composer_delta"), dtype=np.float32)
        influence = np.asarray(meta.get("composer_influence"), dtype=np.float32)
        layer_objs = meta.get("composer_layers", []) or []
        landform_objs = meta.get("composer_landforms", []) or []
        overlap_count = np.zeros_like(influence, dtype=np.uint8)
        for layer in [*layer_objs, *landform_objs]:
            overlap_count += (np.asarray(layer.influence) > 0.05).astype(np.uint8)
        report["composer"] = {
            "donor_count": len(layer_objs),
            "landform_count": len(landform_objs),
            "coverage_fraction": float(np.mean(influence > 0.05)),
            "overlap_fraction": float(np.mean(overlap_count >= 2)),
            "delta_rms_m": float(np.sqrt(np.mean(delta * delta))),
            "delta_p95_abs_m": float(np.percentile(np.abs(delta), 95)),
            "delta_max_abs_m": float(np.max(np.abs(delta))),
            "layers": [
                {
                    "name": layer.name,
                    "source": Path(layer.source_path).name,
                    "source_shape": list(layer.source_shape),
                    "coverage_fraction": float(np.mean(np.asarray(layer.influence) > 0.05)),
                    "delta_p95_abs_m": float(np.percentile(
                        np.abs(np.asarray(layer.contribution_m))[np.asarray(layer.influence) > 0.05], 95
                    )) if np.any(np.asarray(layer.influence) > 0.05) else 0.0,
                }
                for layer in layer_objs
            ],
            "landforms": [
                {
                    "name": layer.name,
                    "kind": layer.kind,
                    "source": Path(layer.source_path).name,
                    "source_shape": list(layer.source_shape),
                    "placement_mode": layer.placement_mode,
                    "library_selected": layer.library_selected,
                    "resolved_center": list(layer.resolved_center),
                    "resolved_scale": list(layer.resolved_scale),
                    "resolved_rotation_deg": layer.resolved_rotation_deg,
                    "resolved_river_t": layer.resolved_river_t,
                    "resolved_river_offset_m": layer.resolved_river_offset_m,
                    "placement_hydrology_score": layer.placement_hydrology_score,
                    "coverage_fraction": float(np.mean(np.asarray(layer.influence) > 0.05)),
                    "delta_p95_abs_m": float(np.percentile(
                        np.abs(np.asarray(layer.contribution_m))[np.asarray(layer.influence) > 0.05], 95
                    )) if np.any(np.asarray(layer.influence) > 0.05) else 0.0,
                }
                for layer in landform_objs
            ],
        }

    if cfg.splines.enabled:
        spline_layers = meta.get("semantic_splines", []) or []
        delta = np.asarray(meta.get("semantic_spline_delta"), dtype=np.float32)
        influence = np.asarray(meta.get("semantic_spline_influence"), dtype=np.float32)
        report["semantic_splines"] = {
            "count": len(spline_layers),
            "coverage_fraction": float(np.mean(influence > 0.05)) if influence.ndim == 2 else 0.0,
            "delta_rms_m": float(np.sqrt(np.mean(delta * delta))) if delta.ndim == 2 else 0.0,
            "delta_p95_abs_m": float(np.percentile(np.abs(delta), 95)) if delta.ndim == 2 else 0.0,
            "delta_max_abs_m": float(np.max(np.abs(delta))) if delta.ndim == 2 else 0.0,
            "layers": [
                {
                    "name": layer.name,
                    "kind": layer.kind,
                    "mode": layer.mode,
                    "source": Path(layer.source_path).name if layer.source_path else "",
                    "library_selected": bool(layer.library_selected),
                    "resolved_strength_factor": float(layer.resolved_strength_factor),
                    "hydrology_score_before": float(layer.hydrology_score_before),
                    "hydrology_score_after": float(layer.hydrology_score_after),
                    "coverage_fraction": float(layer.coverage_fraction),
                    "delta_p95_abs_m": float(layer.delta_p95_abs_m),
                    "delta_max_abs_m": float(layer.delta_max_abs_m),
                }
                for layer in spline_layers
            ],
        }

    if hydrology is not None:
        report["hydrology"] = hydrology.report
    if production is not None:
        report["cs2_production"] = {
            "water": production["water"],
            "world_center_max_abs_error_m": production.get("world_center_max_abs_error_m", 0.0),
            "vertical_encoding": production["manifest"]["vertical_encoding"],
        }

    out = Path(out_dir)
    with open(out / "qa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    shutil.copy2(recipe, out / "recipe.yaml")

    print(f"Generated: {cfg.name}")
    print(f"Output: {out.resolve()}")
    print(f"Elevation: {report['elevation_min_m']:.1f}–{report['elevation_max_m']:.1f} m")
    print(f"Relief: {report['total_relief_m']:.1f} m")
    print(f"<= {effective_grade_cutoff_percent(cfg):g}% grade: {report['fraction_under_target_grade']*100:.1f}%")
    if cfg.composer.enabled:
        print(
            f"Composer: {report['composer_donor_count']} donor layer(s), "
            f"{report['composer_landform_count']} semantic landform(s); "
            f"max |delta| {report['composer_max_abs_delta_m']:.1f} m"
        )
    if cfg.splines.enabled:
        srep = report.get("semantic_splines", {})
        print(
            f"Semantic splines: {srep.get('count', 0)} layer(s); "
            f"coverage {srep.get('coverage_fraction', 0.0)*100:.1f}%; "
            f"max |delta| {srep.get('delta_max_abs_m', 0.0):.1f} m"
        )
    if hydrology is not None:
        h = hydrology.report
        print(
            "Hydrology: "
            f"{h['raw_interior_sinks']} raw sinks -> {h['conditioned_interior_sinks']} after conditioning; "
            f"largest outlet {h['largest_boundary_accumulation_km2']:.2f} km²"
        )
        if cfg.river.enabled:
            print(
                "Semantic river: "
                f"connected={h['semantic_river_connected']} "
                f"valid_outlet={h['semantic_river_valid_outlet']}"
            )
    return 0



def _parse_crop(value: str) -> list[float]:
    vals = [float(v.strip()) for v in value.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1")
    x0, y0, x1, y1 = vals
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise argparse.ArgumentTypeError("crop values must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1")
    return vals


def _crop_array(arr: np.ndarray, crop: list[float]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    h, w = arr.shape
    c0 = int(np.floor(x0 * (w - 1)))
    c1 = int(np.ceil(x1 * (w - 1))) + 1
    r0 = int(np.floor(y0 * (h - 1)))
    r1 = int(np.ceil(y1 * (h - 1))) + 1
    return arr[r0:min(r1, h), c0:min(c1, w)].astype(np.float32, copy=True)


def _save_asset_previews(asset, out_asset: Path) -> None:
    preview_dir = out_asset.parent / f"{out_asset.stem}_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mask = np.clip(asset.mask, 0.0, 1.0)
    Image.fromarray(np.rint(mask * 255).astype(np.uint8)).save(preview_dir / "semantic_mask.png")
    weighted = asset.relief_m * asset.mask
    scale = max(float(np.percentile(np.abs(weighted), 99)), 0.25)
    relief_view = np.clip(weighted / scale, -1, 1) * 0.5 + 0.5
    Image.fromarray(np.rint(relief_view * 255).astype(np.uint8)).save(preview_dir / "relative_relief.png")
    (preview_dir / "metadata.json").write_text(json.dumps(asset.metadata, indent=2), encoding="utf-8")


def cmd_extract_landform(
    input_path: str,
    kind: str,
    name: str,
    crop: list[float],
    base_level_m: float,
    elevation_scale_m: float,
    feather_fraction: float,
    out_path: str,
) -> int:
    layer = DonorLayerConfig(
        source=input_path,
        source_base_level_m=base_level_m,
        source_elevation_scale_m=elevation_scale_m,
    )
    z, source = load_donor(layer)
    selected = _crop_array(z, crop)
    asset = extract_landform(
        selected,
        kind,
        name,
        source_path=str(source),
        source_crop=crop,
        feather_fraction=feather_fraction,
    )
    out = save_landform_asset(asset, out_path)
    _save_asset_previews(asset, out)
    print(f"Extracted {asset.kind}: {asset.name}")
    print(f"Asset: {out.resolve()}")
    print(f"Mask coverage (>5%): {asset.metadata['mask_coverage_fraction_005']*100:.1f}%")
    print(f"Relative relief: {asset.metadata['relief_min_m']:.1f} to {asset.metadata['relief_max_m']:.1f} m")
    return 0


def cmd_detect_landforms(
    input_path: str,
    kind: str,
    base_level_m: float,
    elevation_scale_m: float,
    max_candidates: int,
) -> int:
    layer = DonorLayerConfig(
        source=input_path,
        source_base_level_m=base_level_m,
        source_elevation_scale_m=elevation_scale_m,
    )
    z, source = load_donor(layer)
    candidates = detect_landform_candidates(z, kind, max_candidates=max_candidates)
    payload = {"source": str(source), "kind": kind, "candidates": candidates}
    print(json.dumps(payload, indent=2))
    return 0



def cmd_index_landforms(directory: str, out_path: str | None) -> int:
    out = write_landform_index(directory, out_path)
    entries = load_landform_index(out)
    print(f"Indexed {len(entries)} landform asset(s)")
    print(f"Index: {out.resolve()}")
    return 0


def cmd_search_landforms(
    library: str,
    kind: str,
    name_contains: str,
    tags: list[str],
    min_relief_m: float,
    max_relief_m: float,
    limit: int,
) -> int:
    entries = load_landform_index(library)
    results = search_landforms(
        entries,
        kind=kind,
        name_contains=name_contains,
        tags=tags,
        min_relief_m=min_relief_m,
        max_relief_m=max_relief_m,
        limit=limit,
    )
    payload = [
        {
            "name": e.name,
            "kind": e.kind,
            "path": e.path,
            "relief_p95_abs_m": e.relief_p95_abs_m,
            "coverage_fraction": e.coverage_fraction,
            "principal_axis_deg": e.principal_axis_deg,
            "elongation": e.elongation,
            "tags": e.tags,
        }
        for e in results
    ]
    print(json.dumps(payload, indent=2))
    return 0



def cmd_project_create(recipe: str, out_path: str, name: str | None = None) -> int:
    cfg = TerrainConfig.from_yaml(recipe)
    project = TerrainProject.from_legacy_config(cfg, name=name or cfg.name)
    project.render_settings.final_work_resolution = int(cfg.work_resolution)
    project.render_settings.final_export_resolution = int(cfg.export_resolution)
    project.render_settings.preview_work_resolution = max(192, min(384, int(cfg.work_resolution // 2)))
    project.render_settings.preview_export_resolution = max(384, min(1024, int(cfg.export_resolution // 4)))
    out = project.save(out_path)
    guide = project.save_authoring_guide(out.with_suffix(".guide.png"), size=1200)
    print(f"Created editable project: {project.name}")
    print(f"Objects: {len(project.objects)}")
    print(f"Project: {out}")
    print(f"Guide: {guide}")
    return 0


def cmd_project_render(project_path: str, out_dir: str, mode: str) -> int:
    project = TerrainProject.load(project_path)
    result = project.render(mode)
    cfg = result.config
    meta = result.metadata
    hydro = result.hydrology
    exported = export_outputs(result.elevation_m, meta["water_mask"], cfg, out_dir)
    if cfg.composer.enabled:
        export_composer_outputs(meta, cfg, out_dir)
    if cfg.geomorphology.enabled:
        export_geomorphology_outputs(meta, cfg, out_dir)
    if cfg.splines.enabled:
        export_spline_outputs(meta, cfg, out_dir)
    if hydro is not None:
        export_hydrology_outputs(hydro, cfg, out_dir)
    report = terrain_report(exported.get("diagnostic_elevation", exported["elevation_export"]), cfg.extent_km, effective_grade_cutoff_percent(cfg))
    report.update({
        "project": project.name,
        "schema_version": project.schema_version,
        "render_mode": mode,
        "object_count": len(project.objects),
        "history_depth": len(project.history),
        "semantic_spline_count": len(meta.get("semantic_splines", [])),
        "water_fraction": float(np.mean(meta["water_mask"])),
        "bible_profile": validate_bible_profile(cfg, project.objects),
    })
    if hydro is not None:
        report["hydrology"] = hydro.report
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report["objects"] = [o.to_dict() for o in project.objects]
    report["last_dirty_bounds"] = list(project.last_dirty_bounds()) if project.last_dirty_bounds() else None
    (out / "project_render.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(project_path, out / "project.ftsproject.json")
    project.save_authoring_guide(out / "authoring_objects.png", size=1200)
    print(f"Rendered project: {project.name} ({mode})")
    print(f"Objects: {len(project.objects)} | work={cfg.work_resolution} | export={cfg.export_resolution}")
    print(f"<= {effective_grade_cutoff_percent(cfg):g}% grade: {report['fraction_under_target_grade']*100:.1f}%")
    if hydro is not None:
        print(f"Hydrology largest outlet: {hydro.report['largest_boundary_accumulation_km2']:.2f} km²")
    return 0


def cmd_project_summary(project_path: str) -> int:
    project = TerrainProject.load(project_path)
    payload = {
        "name": project.name,
        "schema_version": project.schema_version,
        "objects": [o.to_dict() for o in project.objects],
        "render_settings": {
            "preview_work_resolution": project.render_settings.preview_work_resolution,
            "preview_export_resolution": project.render_settings.preview_export_resolution,
            "final_work_resolution": project.render_settings.final_work_resolution,
            "final_export_resolution": project.render_settings.final_export_resolution,
        },
        "history_depth": len(project.history),
        "redo_depth": len(project.redo_stack),
        "last_dirty_bounds": list(project.last_dirty_bounds()) if project.last_dirty_bounds() else None,
    }
    print(json.dumps(payload, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fts", description="Fictional Terrain Studio core prototype")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate terrain from a YAML recipe")
    g.add_argument("recipe")
    g.add_argument("--out", required=True, dest="out_dir")

    e = sub.add_parser("extract-landform", help="Extract a reusable semantic landform asset")
    e.add_argument("input")
    e.add_argument("--kind", required=True, choices=["valley", "ridge", "terrace", "bluff"])
    e.add_argument("--name", required=True)
    e.add_argument("--crop", type=_parse_crop, default=[0.0, 0.0, 1.0, 1.0])
    e.add_argument("--base-level", type=float, default=0.0, dest="base_level_m")
    e.add_argument("--elevation-scale", type=float, default=1024.0, dest="elevation_scale_m")
    e.add_argument("--feather", type=float, default=0.045, dest="feather_fraction")
    e.add_argument("--out", required=True, dest="out_path")

    d = sub.add_parser("detect-landforms", help="Suggest semantic landform candidate regions")
    d.add_argument("input")
    d.add_argument("--kind", required=True, choices=["valley", "ridge", "terrace", "bluff"])
    d.add_argument("--base-level", type=float, default=0.0, dest="base_level_m")
    d.add_argument("--elevation-scale", type=float, default=1024.0, dest="elevation_scale_m")
    d.add_argument("--max", type=int, default=5, dest="max_candidates")

    li = sub.add_parser("index-landforms", help="Build a portable searchable landform-library index")
    li.add_argument("directory")
    li.add_argument("--out", dest="out_path")

    ls = sub.add_parser("search-landforms", help="Search a landform library")
    ls.add_argument("library")
    ls.add_argument("--kind", default="")
    ls.add_argument("--name", default="", dest="name_contains")
    ls.add_argument("--tag", action="append", default=[], dest="tags")
    ls.add_argument("--min-relief", type=float, default=0.0, dest="min_relief_m")
    ls.add_argument("--max-relief", type=float, default=1.0e9, dest="max_relief_m")
    ls.add_argument("--limit", type=int, default=10)

    pc = sub.add_parser("project-create", help="Migrate a YAML recipe into an editable .ftsproject.json project")
    pc.add_argument("recipe")
    pc.add_argument("--name", default=None)
    pc.add_argument("--out", required=True, dest="out_path")

    pr = sub.add_parser("project-render", help="Render an editable .ftsproject.json project")
    pr.add_argument("project")
    pr.add_argument("--mode", choices=["preview", "final"], default="preview")
    pr.add_argument("--out", required=True, dest="out_dir")

    ps = sub.add_parser("project-summary", help="Print project objects and render settings")
    ps.add_argument("project")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        return cmd_generate(args.recipe, args.out_dir)
    if args.command == "extract-landform":
        return cmd_extract_landform(
            args.input, args.kind, args.name, args.crop, args.base_level_m,
            args.elevation_scale_m, args.feather_fraction, args.out_path
        )
    if args.command == "detect-landforms":
        return cmd_detect_landforms(
            args.input, args.kind, args.base_level_m, args.elevation_scale_m, args.max_candidates
        )
    if args.command == "index-landforms":
        return cmd_index_landforms(args.directory, args.out_path)
    if args.command == "search-landforms":
        return cmd_search_landforms(
            args.library, args.kind, args.name_contains, args.tags,
            args.min_relief_m, args.max_relief_m, args.limit
        )
    if args.command == "project-create":
        return cmd_project_create(args.recipe, args.out_path, args.name)
    if args.command == "project-render":
        return cmd_project_render(args.project, args.out_dir, args.mode)
    if args.command == "project-summary":
        return cmd_project_summary(args.project)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
