from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil
import uuid
from typing import Any

import numpy as np
from PIL import Image

from .analysis import slope_percent, terrain_report
from .design_profile import effective_grade_cutoff_percent, validate_bible_profile
from .export import (
    export_composer_outputs,
    export_geomorphology_outputs,
    export_hydrology_outputs,
    export_outputs,
    export_spline_outputs,
    hillshade,
    resize_bicubic,
)
from .project import AuthoringObject, ProjectRenderResult, TerrainProject
from .cs2_production import apply_water_terrain, bathymetry_guide, export_cs2_production
from .hydrology import analyze_hydrology
from .measurements import TerrainProfile, infer_profile_role, sample_spline_profile, sample_transect
from .local_edits import apply_local_edits
from .gis_interchange import (
    export_gis_package as write_gis_package,
    import_objects_geojson,
    read_geotiff_to_grid,
    install_heightmaps as install_cs2_heightmaps,
)


@dataclass(slots=True)
class PreviewImage:
    image: Image.Image
    mode: str
    report: dict[str, Any]


class EditorController:
    """Toolkit-agnostic controller used by the v0.9 desktop editor.

    The GUI intentionally owns no terrain-generation logic. This class exposes
    the same project operations that a future Qt client, MCP server, or local
    model agent can call.
    """

    def __init__(self) -> None:
        self.project: TerrainProject | None = None
        self.project_path: Path | None = None
        self.selected_object_id: str | None = None
        self.selected_point_index: int | None = None
        self.last_render: ProjectRenderResult | None = None
        self.last_report: dict[str, Any] | None = None
        self._last_render_digest: str | None = None
        self._last_base_digest: str | None = None
        self._saved_digest: str | None = None

    @property
    def has_project(self) -> bool:
        return self.project is not None

    def open_project(self, path: str | Path) -> TerrainProject:
        p = Path(path).expanduser().resolve()
        self.project = TerrainProject.load(p)
        self.project_path = p
        self.selected_object_id = self.project.objects[0].id if self.project.objects else None
        self.selected_point_index = None
        self.last_render = None
        self.last_report = None
        self._last_render_digest = None
        self._last_base_digest = None
        self._saved_digest = self._project_digest()
        return self.project

    def save_project(self, path: str | Path | None = None) -> Path:
        if self.project is None:
            raise RuntimeError("No project is open")
        target = Path(path).expanduser().resolve() if path is not None else self.project_path
        if target is None:
            raise ValueError("A save path is required for a new project")
        saved = self.project.save(target)
        self.project_path = saved
        self._saved_digest = self._project_digest()
        self.remove_autosave()
        return saved


    def _project_digest(self) -> str | None:
        if self.project is None:
            return None
        raw = json.dumps(self.project.to_dict(include_history=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _base_project_digest(self) -> str | None:
        """Digest terrain inputs excluding v0.17 local correction objects.

        When this digest is unchanged, the expensive procedural/GIS base terrain
        is still valid and preview can be rebuilt by reapplying only the local
        correction stack.
        """
        if self.project is None:
            return None
        payload = self.project.to_dict(include_history=False)
        payload["objects"] = [
            o for o in payload.get("objects", [])
            if str(o.get("object_type", "")).strip().lower().replace("-", "_") != "local_edit"
        ]
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def can_incremental_preview(self) -> bool:
        if self.project is None or self.last_render is None:
            return False
        if self._last_base_digest is None or self._base_project_digest() != self._last_base_digest:
            return False
        base = self.last_render.metadata.get("base_elevation_before_local_edits")
        return isinstance(base, np.ndarray) and base.shape == self.last_render.elevation_m.shape

    @property
    def is_dirty(self) -> bool:
        if self.project is None:
            return False
        return self._project_digest() != self._saved_digest

    @property
    def preview_is_stale(self) -> bool:
        if self.project is None or self.last_render is None:
            return True
        return self._project_digest() != self._last_render_digest

    def autosave_path(self) -> Path | None:
        if self.project_path is None:
            return None
        p = self.project_path
        return p.with_name(f".{p.name}.autosave")

    def write_autosave(self) -> Path | None:
        if self.project is None or not self.is_dirty:
            return None
        target = self.autosave_path()
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.project.to_dict(include_history=True), indent=2), encoding="utf-8")
        return target

    def remove_autosave(self) -> None:
        p = self.autosave_path()
        if p is not None:
            p.unlink(missing_ok=True)

    @staticmethod
    def recovery_path(project_path: str | Path) -> Path:
        p = Path(project_path).expanduser().resolve()
        return p.with_name(f".{p.name}.autosave")

    @classmethod
    def recovery_available(cls, project_path: str | Path) -> bool:
        p = Path(project_path).expanduser().resolve()
        a = cls.recovery_path(p)
        if not a.exists():
            return False
        try:
            return (not p.exists()) or a.stat().st_mtime > p.stat().st_mtime
        except OSError:
            return True

    def recover_autosave(self, project_path: str | Path) -> TerrainProject:
        original = Path(project_path).expanduser().resolve()
        auto = self.recovery_path(original)
        if not auto.exists():
            raise FileNotFoundError(auto)
        self.project = TerrainProject.load(auto)
        self.project.project_dir = str(original.parent)
        self.project_path = original
        self.selected_object_id = self.project.objects[0].id if self.project.objects else None
        self.selected_point_index = None
        self.last_render = None
        self.last_report = None
        self._last_render_digest = None
        self._last_base_digest = None
        # Keep the digest anchored to the on-disk original so the recovered
        # project remains dirty until the user explicitly saves it.
        if original.exists():
            disk_project = TerrainProject.load(original)
            raw = json.dumps(disk_project.to_dict(include_history=False), sort_keys=True, separators=(",", ":"))
            self._saved_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        else:
            self._saved_digest = None
        return self.project

    def clone_project(self) -> TerrainProject:
        if self.project is None:
            raise RuntimeError("No project is open")
        raw = self.project.to_dict(include_history=True)
        clone = TerrainProject(
            name=raw["name"],
            base_config=deepcopy(raw["base_config"]),
            objects=[AuthoringObject.from_dict(v) for v in raw.get("objects", [])],
            render_settings=deepcopy(self.project.render_settings),
            history=deepcopy(self.project.history),
            redo_stack=deepcopy(self.project.redo_stack),
            schema_version=str(raw.get("schema_version", self.project.schema_version)),
            project_dir=self.project.project_dir,
        )
        return clone

    def select_object(self, object_id: str | None) -> AuthoringObject | None:
        if self.project is None or object_id is None:
            self.selected_object_id = None
            self.selected_point_index = None
            return None
        obj = self.project.get_object(object_id)
        self.selected_object_id = obj.id
        self.selected_point_index = None
        return obj

    def selected_object(self) -> AuthoringObject | None:
        if self.project is None or self.selected_object_id is None:
            return None
        try:
            return self.project.get_object(self.selected_object_id)
        except KeyError:
            self.selected_object_id = None
            self.selected_point_index = None
            return None

    def set_selected_point(self, point_index: int | None) -> None:
        obj = self.selected_object()
        if point_index is None:
            self.selected_point_index = None
            return
        if obj is None or point_index < 0 or point_index >= len(obj.control_points):
            raise IndexError(point_index)
        self.selected_point_index = int(point_index)

    def move_point(self, object_id: str, point_index: int, x: float, y: float) -> None:
        if self.project is None:
            raise RuntimeError("No project is open")
        self.project.move_spline_point(object_id, point_index, float(x), float(y))
        self.selected_object_id = object_id
        self.selected_point_index = point_index

    def insert_point(self, object_id: str, point_index: int, x: float, y: float) -> None:
        if self.project is None:
            raise RuntimeError("No project is open")
        self.project.insert_spline_point(object_id, point_index, float(x), float(y))
        self.selected_object_id = object_id
        self.selected_point_index = point_index

    def delete_point(self, object_id: str, point_index: int) -> None:
        if self.project is None:
            raise RuntimeError("No project is open")
        self.project.delete_spline_point(object_id, point_index)
        self.selected_object_id = object_id
        obj = self.project.get_object(object_id)
        self.selected_point_index = min(point_index, len(obj.control_points) - 1)

    def update_selected_object(
        self,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        properties: dict[str, Any] | None = None,
        replace_properties: bool = False,
    ) -> None:
        if self.project is None or self.selected_object_id is None:
            raise RuntimeError("No object is selected")
        self.project.update_object(
            self.selected_object_id,
            name=name,
            enabled=enabled,
            properties=properties,
            replace_properties=replace_properties,
            label="Inspector edit",
        )

    def add_template_object(self, kind: str) -> str:
        if self.project is None:
            raise RuntimeError("No project is open")
        kind = kind.strip().lower().replace("-", "_")
        templates: dict[str, tuple[str, str, list[tuple[float, float]], dict[str, Any]]] = {
            "river": (
                "river", "New River", [(-0.55, 0.65), (-0.10, 0.18), (0.48, -0.42)],
                {
                    "role": "tributary", "channel_enabled": True,
                    "valley_width_m": 720.0, "floodplain_width_m": 300.0,
                    "channel_width_m": 85.0, "valley_depth_m": 8.0,
                    "channel_depth_m": 3.0, "longitudinal_drop_m": 7.0,
                    "weight": 0.75, "optimize_hydrology": True,
                },
            ),
            "valley": (
                "spline", "New Valley", [(-0.65, 0.35), (0.0, 0.0), (0.58, -0.30)],
                {"kind": "valley", "mode": "procedural", "width_m": 900.0, "strength_m": 8.0, "weight": 0.8},
            ),
            "bluff": (
                "spline", "New Bluff", [(-0.65, 0.20), (0.0, 0.35), (0.65, 0.12)],
                {"kind": "bluff", "mode": "procedural", "width_m": 1000.0, "strength_m": 9.0, "side": "left", "weight": 0.7},
            ),
            "moraine": (
                "spline", "New Moraine", [(-0.70, -0.20), (-0.05, 0.05), (0.72, -0.10)],
                {"kind": "moraine", "mode": "procedural", "width_m": 1300.0, "strength_m": 6.0, "weight": 0.7},
            ),
            "lake": (
                "lake", "New Lake", [(-0.35, 0.30), (0.10, 0.38), (0.35, 0.05), (0.12, -0.28), (-0.28, -0.18)],
                {"depth_m": 9.0, "littoral_width_m": 180.0, "shore_freeboard_m": 0.8, "weight": 1.0},
            ),
            "coast": (
                "coast", "New Coast", [(-1.02, 0.55), (-0.35, 0.38), (0.30, 0.18), (1.02, -0.05)],
                {"water_side": "left", "depth_m": 12.0, "outer_depth_m": 20.0, "littoral_width_m": 260.0, "weight": 1.0},
            ),
            "harbor": (
                "harbor", "New Harbor", [(0.25, -0.10)],
                {"radius_x_m": 420.0, "radius_y_m": 260.0, "depth_m": 10.0, "feather_m": 160.0, "weight": 1.0},
            ),
            "water_source": (
                "water_source", "New Water Source", [(-0.85, 0.45)],
                {
                    "source_type": "border_river", "purpose": "river_inlet",
                    "surface_elevation_m": 150.0, "footprint_width_m": 180.0,
                    "relative_flow_hint": "medium", "simulation_mode": "modern",
                    "linked_object_id": "", "source_role": "PROJECT_CANON",
                    "confidence": "CONFIRMED",
                },
            ),
            "local_edit": (
                "local_edit", "New Local Grade Repair", [(-0.15, 0.05), (0.15, -0.05)],
                {
                    "operation": "grade_corridor", "measurement_role": "general",
                    "width_m": 90.0, "feather_m": 70.0, "strength": 1.0,
                    "max_cut_fill_m": 8.0, "engineering_class": "LOCAL_CORRECTION",
                },
            ),
        }
        if kind not in templates:
            raise ValueError(f"Unknown template: {kind}")
        object_type, name, points, props = templates[kind]
        obj = AuthoringObject.create(object_type, name, points, props)
        self.project.add_object(obj, label=f"Add {kind}")
        self.select_object(obj.id)
        return obj.id

    def export_gis_package(self, out_dir: str | Path, *, render_if_needed: bool = True) -> dict[str, Any]:
        if self.project is None:
            raise RuntimeError("No project is open")
        if self.last_render is None and render_if_needed:
            self.render_preview(hydrology=False)
        if self.last_render is None:
            raise RuntimeError("Render the project before exporting a GIS package")
        result = self.last_render
        water = np.asarray(result.metadata.get("water_mask"), dtype=bool)
        prod_elev, prod_water, _ = apply_water_terrain(result.elevation_m, water, result.config, objects=self.project.objects)
        depth = np.where(prod_water, np.maximum(0.0, np.percentile(prod_elev[prod_water], 85) - prod_elev), 0.0).astype(np.float32) if np.any(prod_water) else np.zeros_like(prod_elev, dtype=np.float32)
        return write_gis_package(prod_elev, self.project.objects, result.config, out_dir, water_depth_m=depth)

    def import_gis_objects(
        self,
        path: str | Path,
        *,
        update_existing: bool = False,
        allow_hydro_updates: bool = False,
    ) -> dict[str, Any]:
        if self.project is None:
            raise RuntimeError("No project is open")
        cfg = self.project.compile_config("preview")
        items, report = import_objects_geojson(path, cfg, preserve_ids=True)
        imported = 0
        updated = 0
        skipped = 0
        warnings = list(report.get("warnings", []))
        hydro_types = {"river", "lake", "coast", "sea", "shoreline"}
        locked = bool(getattr(cfg.bible_profile, "enabled", False) and getattr(cfg.bible_profile, "hydrography_locked", False))
        for raw in items:
            obj_id = raw.get("id")
            typ = str(raw["object_type"]).lower().replace("-", "_")
            existing = None
            if obj_id:
                try:
                    existing = self.project.get_object(str(obj_id))
                except KeyError:
                    existing = None
            if existing is not None:
                if not update_existing:
                    skipped += 1
                    warnings.append(f"Skipped existing object {obj_id}; enable update_existing for an explicit GIS round-trip update.")
                    continue
                if locked and typ in hydro_types and not allow_hydro_updates:
                    skipped += 1
                    warnings.append(f"Skipped locked hydrography update for {obj_id}; explicit allow_hydro_updates is required.")
                    continue
                self.project.update_object(
                    str(obj_id),
                    name=raw["name"], enabled=bool(raw["enabled"]),
                    control_points=raw["control_points"], properties=raw["properties"],
                    replace_properties=True, label="GIS round-trip update",
                )
                updated += 1
                continue
            props = dict(raw["properties"])
            enabled = bool(raw["enabled"])
            if locked and typ in hydro_types and not allow_hydro_updates:
                enabled = False
                props["gis_review_required"] = True
                warnings.append(f"Imported {raw['name']} disabled because hydrography is locked; review before enabling.")
            obj = AuthoringObject.create(typ, raw["name"], raw["control_points"], props, object_id=str(obj_id) if obj_id else None)
            obj.enabled = enabled
            self.project.add_object(obj, label="Import GIS object")
            imported += 1
        self.last_render = None
        self.last_report = None
        self._last_render_digest = None
        return {**report, "imported": imported, "updated": updated, "skipped": skipped, "warnings": warnings}

    def set_gis_base_dem(self, path: str | Path, *, resampling: str = "bilinear") -> dict[str, Any]:
        if self.project is None:
            raise RuntimeError("No project is open")
        cfg = self.project.compile_config("preview")
        source = Path(path).expanduser().resolve()
        # Validate georeferencing and overlap before mutating project state.
        _, meta = read_geotiff_to_grid(source, cfg, resolution=max(128, min(256, cfg.work_resolution)), resampling=resampling)
        gis = deepcopy(self.project.base_config.get("gis", {}) or {})
        try:
            rel = source.relative_to(Path(self.project.project_dir).resolve())
            stored = str(rel)
        except Exception:
            stored = str(source)
        gis.update({
            "enabled": True, "base_dem_path": stored, "base_dem_mode": "replace",
            "base_dem_resampling": str(resampling), "source_authority": "CARTO/GIS VERIFIED",
        })
        self.project.base_config["gis"] = gis
        self.last_render = None
        self.last_report = None
        self._last_render_digest = None
        return {"base_dem_path": stored, "mode": "replace", "validation": meta}

    def clear_gis_base_dem(self) -> None:
        if self.project is None:
            raise RuntimeError("No project is open")
        gis = deepcopy(self.project.base_config.get("gis", {}) or {})
        gis["base_dem_path"] = ""
        gis["base_dem_mode"] = "procedural"
        self.project.base_config["gis"] = gis
        self.last_render = None
        self.last_report = None
        self._last_render_digest = None

    def install_heightmaps(self, production_dir: str | Path, target_dir: str | Path | None = None, *, prefix: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        name = prefix or (self.project.name if self.project is not None else "FTS")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "FTS"
        return install_cs2_heightmaps(production_dir, target_dir, prefix=safe, overwrite=overwrite)

    def delete_selected_object(self) -> bool:
        if self.project is None or self.selected_object_id is None:
            return False
        obj_id = self.selected_object_id
        self.project.delete_object(obj_id)
        self.selected_object_id = self.project.objects[0].id if self.project.objects else None
        self.selected_point_index = None
        return True

    def undo(self) -> bool:
        if self.project is None:
            return False
        ok = self.project.undo()
        self._repair_selection()
        return ok

    def redo(self) -> bool:
        if self.project is None:
            return False
        ok = self.project.redo()
        self._repair_selection()
        return ok

    def _repair_selection(self) -> None:
        if self.project is None:
            self.selected_object_id = None
            self.selected_point_index = None
            return
        ids = {o.id for o in self.project.objects}
        if self.selected_object_id not in ids:
            self.selected_object_id = self.project.objects[0].id if self.project.objects else None
            self.selected_point_index = None
        elif self.selected_point_index is not None:
            obj = self.selected_object()
            if obj is None or self.selected_point_index >= len(obj.control_points):
                self.selected_point_index = None

    def _preview_report(self, result: ProjectRenderResult, *, incremental: bool = False) -> dict[str, Any]:
        if self.project is None:
            raise RuntimeError("No project is open")
        report = terrain_report(
            result.elevation_m,
            result.config.extent_km,
            effective_grade_cutoff_percent(result.config),
        )
        if result.hydrology is not None:
            report["hydrology"] = deepcopy(result.hydrology.report)
        report["object_count"] = len(self.project.objects)
        report["history_depth"] = len(self.project.history)
        report["render_mode"] = "preview"
        report["preview_reused_base"] = bool(incremental)
        report["local_edit_count"] = int(result.metadata.get("local_edit_count", 0))
        report["local_edits"] = deepcopy(result.metadata.get("local_edits", []))
        report["bible_profile"] = validate_bible_profile(result.config, self.project.objects)
        return report

    def render_preview(self, *, hydrology: bool = True, progress=None, cancel_check=None) -> ProjectRenderResult:
        if self.project is None:
            raise RuntimeError("No project is open")
        result = self.project.render("preview", hydrology=hydrology, progress=progress, cancel_check=cancel_check)
        self.last_render = result
        self._last_render_digest = self._project_digest()
        self._last_base_digest = self._base_project_digest()
        self.last_report = self._preview_report(result, incremental=False)
        return result

    def render_preview_incremental(self, *, hydrology: bool = True) -> ProjectRenderResult:
        """Reapply only local correction objects over a cached base preview.

        This is intentionally conservative: any change to rivers, landforms, GIS
        base terrain, render settings, or other semantic objects changes the base
        digest and forces the normal full preview path.
        """
        if self.project is None or self.last_render is None or not self.can_incremental_preview:
            return self.render_preview(hydrology=hydrology)
        cfg = self.project.compile_config("preview")
        base = np.asarray(self.last_render.metadata["base_elevation_before_local_edits"], dtype=np.float32)
        elevation, local_edit_report = apply_local_edits(base, cfg.extent_km, self.project.objects)
        metadata = deepcopy(self.last_render.metadata)
        metadata["base_elevation_before_local_edits"] = base.copy()
        metadata["local_edits"] = local_edit_report
        metadata["local_edit_count"] = len(local_edit_report)
        metadata["incremental_preview"] = True
        hydro = None
        if hydrology and cfg.hydrology.enabled:
            hydro = analyze_hydrology(
                elevation,
                metadata["water_mask"],
                cfg.extent_km,
                stream_threshold_km2=cfg.hydrology.stream_threshold_km2,
                floodplain_hand_m=cfg.hydrology.floodplain_hand_m,
                floodplain_slope_percent=cfg.hydrology.floodplain_slope_percent,
            )
        result = ProjectRenderResult(cfg, elevation, metadata, hydro, "preview")
        self.last_render = result
        self._last_render_digest = self._project_digest()
        self._last_base_digest = self._base_project_digest()
        self.last_report = self._preview_report(result, incremental=True)
        return result

    def accept_preview_result(self, result: ProjectRenderResult, report: dict[str, Any]) -> None:
        """Install a preview rendered by a worker thread as the controller's current surface."""
        self.last_render = result
        self.last_report = deepcopy(report)
        self._last_render_digest = self._project_digest()
        self._last_base_digest = self._base_project_digest()

    def preview_image(self, view: str = "hillshade", max_size: int = 1000) -> PreviewImage:
        if self.last_render is None:
            self.render_preview()
        assert self.last_render is not None
        result = self.last_render
        z = np.asarray(result.elevation_m, dtype=np.float32)
        target = min(max(int(max_size), 256), 1200)
        if z.shape != (target, target):
            z = resize_bicubic(z, target)
        key = view.strip().lower()
        if key == "elevation":
            mn, mx = float(np.min(z)), float(np.max(z))
            arr = (z - mn) / max(mx - mn, 1e-6)
        elif key == "slope":
            s = slope_percent(z, result.config.extent_km)
            cutoff = effective_grade_cutoff_percent(result.config)
            arr = np.clip(s / max(cutoff * 2.0, 1.0), 0.0, 1.0)
        elif key == "buildability":
            s = slope_percent(z, result.config.extent_km)
            arr = (s <= effective_grade_cutoff_percent(result.config)).astype(np.float32)
        elif key == "flow" and result.hydrology is not None:
            flow = np.log1p(np.asarray(result.hydrology.accumulation_km2, dtype=np.float32))
            flow = resize_bicubic(flow, target)
            arr = flow / max(float(np.max(flow)), 1e-6)
        elif key == "water":
            base_water = np.asarray(result.metadata.get("water_mask"), dtype=bool)
            semantic_elev, semantic_water, _ = apply_water_terrain(
                result.elevation_m, base_water, result.config,
                objects=self.project.objects if self.project is not None else None,
            )
            del semantic_elev
            arr = resize_bicubic(semantic_water.astype(np.float32), target)
        elif key == "bathymetry":
            base_water = np.asarray(result.metadata.get("water_mask"), dtype=bool)
            semantic_elev, semantic_water, _ = apply_water_terrain(
                result.elevation_m, base_water, result.config,
                objects=self.project.objects if self.project is not None else None,
            )
            bath = bathymetry_guide(semantic_elev, semantic_water, result.config)
            delta = resize_bicubic(-bath.delta_m, target)
            arr = delta / max(float(np.max(delta)), 1e-6)
        else:
            key = "hillshade"
            hs = hillshade(z, result.config.extent_km)
            lo, hi = np.percentile(hs, [2.0, 98.0])
            arr = np.clip((hs - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            # Keep some mid-gray so spline overlays remain readable while
            # substantially improving relief contrast over raw hillshade.
            arr = 0.12 + arr * 0.82
        image = Image.fromarray(np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB")
        return PreviewImage(image=image, mode=key, report=deepcopy(self.last_report or {}))


    def _measurement_surface(self) -> ProjectRenderResult:
        """Return the most recent preview surface, rendering once if needed."""
        if self.project is None:
            raise RuntimeError("No project is open")
        if self.last_render is None:
            self.render_preview(hydrology=False)
        assert self.last_render is not None
        return self.last_render

    def measure_object_profile(self, object_id: str | None = None, *, spacing_m: float = 20.0) -> tuple[TerrainProfile, dict[str, Any]]:
        """Measure elevation/grade along an editable spline object.

        The operation samples the current preview surface and therefore remains
        fast enough for interactive diagnostics. Press Render Preview after
        geometry/terrain edits before treating the profile as authoritative.
        """
        if self.project is None:
            raise RuntimeError("No project is open")
        oid = object_id or self.selected_object_id
        if not oid:
            raise RuntimeError("Select a spline object first")
        obj = self.project.get_object(oid)
        if len(obj.control_points) < 2:
            raise ValueError(f"{obj.name} does not contain a measurable path")
        result = self._measurement_surface()
        role = infer_profile_role(obj.object_type, obj.properties)
        profile = sample_spline_profile(
            result.elevation_m, result.config.extent_km, obj.control_points,
            spacing_m=spacing_m, source=f"object:{obj.id}", role=role,
        )
        summary = profile.summary()
        summary.update({
            "object_id": obj.id,
            "object_name": obj.name,
            "object_type": obj.object_type,
            "preview_mode": result.mode,
            "preview_may_be_stale": bool(self.preview_is_stale),
        })
        return profile, summary

    def measure_transect(
        self, start: tuple[float, float], end: tuple[float, float], *, spacing_m: float = 20.0, role: str = "general"
    ) -> tuple[TerrainProfile, dict[str, Any]]:
        """Measure an arbitrary map transect in normalized -1..1 coordinates."""
        result = self._measurement_surface()
        profile = sample_transect(
            result.elevation_m, result.config.extent_km, start, end, spacing_m=spacing_m, role=role,
        )
        summary = profile.summary()
        summary.update({
            "start": [float(start[0]), float(start[1])],
            "end": [float(end[0]), float(end[1])],
            "preview_mode": result.mode,
            "preview_may_be_stale": bool(self.preview_is_stale),
        })
        return profile, summary

    def add_measurement_correction(
        self,
        profile: TerrainProfile,
        summary: dict[str, Any],
        *,
        operation: str = "grade_corridor",
    ) -> str:
        """Create an undoable local terrain correction from a measured path."""
        if self.project is None:
            raise RuntimeError("No project is open")
        op = str(operation).strip().lower().replace("-", "_")
        if op not in {"grade_corridor", "smooth_corridor"}:
            raise ValueError("operation must be grade_corridor or smooth_corridor")
        role = str(summary.get("role", getattr(profile, "role", "general")) or "general").lower()
        default_width = {"rail": 55.0, "road": 80.0, "river": 120.0, "general": 90.0}.get(role, 90.0)
        # Retain enough points for curved measured objects while keeping local
        # correction geometry light and human-editable.
        count = min(14, max(2, len(profile.x)))
        idx = np.unique(np.linspace(0, len(profile.x) - 1, count).round().astype(int))
        points = [(float(profile.x[i]), float(profile.y[i])) for i in idx]
        props: dict[str, Any] = {
            "operation": op,
            "measurement_role": role,
            "width_m": default_width,
            "feather_m": default_width * 0.75,
            "strength": 1.0 if op == "grade_corridor" else 0.65,
            "max_cut_fill_m": 8.0 if role != "river" else 4.0,
            "engineering_class": "LOCAL_CORRECTION",
            "source_measurement": {
                "distance_m": float(summary.get("distance_m", 0.0)),
                "qa_status": str((summary.get("qa") or {}).get("status", "")),
                "source": str(getattr(profile, "source", "measurement")),
            },
        }
        if op == "smooth_corridor":
            props["sigma_m"] = max(default_width * 0.18, 12.0)
        label = "Grade Repair" if op == "grade_corridor" else "Corridor Smooth"
        obj = AuthoringObject.create("local_edit", f"{label} — {role.title()}", points, props)
        self.project.add_object(obj, label=f"Add {label.lower()}")
        self.select_object(obj.id)
        return obj.id

    def bake_final(self, out_dir: str | Path, *, progress=None, cancel_check=None) -> dict[str, Any]:
        if self.project is None:
            raise RuntimeError("No project is open")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        staging = out / f".fts-incomplete-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=False)

        def checkpoint(stage: str, fraction: float, message: str = "") -> None:
            if cancel_check is not None:
                cancel_check()
            if progress is not None:
                progress(stage, fraction, message)

        try:
            checkpoint("final_render", 0.01, "Starting final terrain bake")
            result = self.project.render("final", hydrology=True, progress=progress, cancel_check=cancel_check)
            cfg = result.config
            meta = result.metadata
            hydro = result.hydrology
            checkpoint("export_heightmap", 0.92, "Writing 4096×4096 16-bit heightmap")
            exported = export_outputs(result.elevation_m, meta["water_mask"], cfg, staging, write_heightmap=not cfg.production.enabled)
            production = None
            if cfg.production.enabled:
                checkpoint("cs2_production", 0.94, "Building CS2 world map, bathymetry, and import manifest")
                production = export_cs2_production(result.elevation_m, meta["water_mask"], cfg, staging, objects=self.project.objects)
            checkpoint("export_diagnostics", 0.95, "Writing diagnostic layers")
            if cfg.composer.enabled:
                export_composer_outputs(meta, cfg, staging)
                checkpoint("export_diagnostics", 0.96, "Writing composer diagnostics")
            if cfg.geomorphology.enabled:
                export_geomorphology_outputs(meta, cfg, staging)
                checkpoint("export_diagnostics", 0.97, "Writing geomorphology diagnostics")
            if cfg.splines.enabled:
                export_spline_outputs(meta, cfg, staging)
                checkpoint("export_diagnostics", 0.98, "Writing spline diagnostics")
            gis_report = None
            if bool(getattr(cfg.gis, "enabled", False)):
                checkpoint("gis_export", 0.982, "Writing canonical QGIS interchange package")
                gis_source = production["production_elevation"] if production is not None else result.elevation_m
                gis_elevation = resize_bicubic(gis_source, 4096)
                gis_report = write_gis_package(gis_elevation, self.project.objects, cfg, staging / "gis")
                if production is not None:
                    production["manifest"]["gis_interchange"] = {
                        "directory": "gis",
                        "terrain": "gis/terrain_m_float32.tif",
                        "semantic_objects": "gis/semantic_objects.geojson",
                        "manifest": "gis/fts_gis_manifest.json",
                    }
                    (staging / "cs2_import_manifest.json").write_text(json.dumps(production["manifest"], indent=2), encoding="utf-8")
            if hydro is not None:
                export_hydrology_outputs(hydro, cfg, staging)
                checkpoint("export_diagnostics", 0.985, "Writing hydrology diagnostics")
            report = terrain_report(
                exported.get("diagnostic_elevation", exported["elevation_export"]),
                cfg.extent_km,
                effective_grade_cutoff_percent(cfg),
            )
            report.update({
                "project": self.project.name,
                "schema_version": self.project.schema_version,
                "render_mode": "final",
                "object_count": len(self.project.objects),
                "history_depth": len(self.project.history),
                "water_fraction": float(np.mean(meta["water_mask"])),
                "cs2_production_enabled": bool(cfg.production.enabled),
                "local_edit_count": int(meta.get("local_edit_count", 0)),
                "local_edits": deepcopy(meta.get("local_edits", [])),
            })
            if production is not None:
                prod_hydro = analyze_hydrology(
                    production["production_elevation"],
                    production["production_water_mask"],
                    cfg.extent_km,
                    fill_epsilon_m=cfg.hydrology.fill_epsilon_m,
                    stream_threshold_km2=cfg.hydrology.stream_threshold_km2,
                    floodplain_hand_m=cfg.hydrology.floodplain_hand_m,
                    floodplain_slope_percent=cfg.hydrology.floodplain_slope_percent,
                    semantic_river_mask=np.asarray(meta["water_mask"], dtype=bool),
                ) if cfg.hydrology.enabled else None
                report["cs2"] = deepcopy(production["manifest"])
                report["cs2_production"] = {
                    "water": deepcopy(production["water"]),
                    "world_center_max_abs_error_m": production.get("world_center_max_abs_error_m", 0.0),
                    "vertical_encoding": deepcopy(production["manifest"]["encoding"]),
                    "validation": deepcopy(production["manifest"]["validation"]),
                    "hydrology": deepcopy(prod_hydro.report) if prod_hydro is not None else None,
                }
            if hydro is not None:
                report["hydrology"] = deepcopy(hydro.report)
            report["bible_profile"] = validate_bible_profile(cfg, self.project.objects)
            if gis_report is not None:
                final_gis_report = deepcopy(gis_report)
                if isinstance(final_gis_report.get("terrain"), dict):
                    final_gis_report["terrain"]["path"] = str(out / "gis" / "terrain_m_float32.tif")
                if isinstance(final_gis_report.get("vectors"), dict):
                    final_gis_report["vectors"]["path"] = str(out / "gis" / "semantic_objects.geojson")
                if final_gis_report.get("water_depth"):
                    final_gis_report["water_depth"]["path"] = str(out / "gis" / "water_depth_m_float32.tif")
                final_gis_report["manifest"] = str(out / "gis" / "fts_gis_manifest.json")
                report["gis_interchange"] = final_gis_report
            report["objects"] = [o.to_dict() for o in self.project.objects]
            report["last_dirty_bounds"] = list(self.project.last_dirty_bounds()) if self.project.last_dirty_bounds() else None
            (staging / "project_render.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (staging / "project.ftsproject.json").write_text(
                json.dumps(self.project.to_dict(include_history=True), indent=2), encoding="utf-8"
            )
            self.project.save_authoring_guide(staging / "authoring_objects.png", size=1200)
            checkpoint("commit_output", 0.995, "Committing completed bake")
            # Only publish files after every stage succeeds. Existing unrelated
            # output files are retained, while same-name outputs are replaced.
            for child in list(staging.iterdir()):
                target = out / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                child.replace(target)
            staging.rmdir()
            checkpoint("complete", 1.0, "Final bake complete")
            self.last_report = report
            return report
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def project_title(self) -> str:
        if self.project is None:
            return "Fictional Terrain Studio"
        name = self.project.name + (" *" if self.is_dirty else "")
        return f"{name} — Fictional Terrain Studio"
