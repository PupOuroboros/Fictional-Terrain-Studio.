from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from threading import RLock
from typing import Any, Callable
import json
import tempfile

import numpy as np

from .app_controller import EditorController
from .library import load_landform_index, search_landforms
from .cs2_production import apply_water_terrain, plan_water_sources, vertical_calibration
from .cs2_workflow import build_editor_assistance

from .design_profile import validate_bible_profile


JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    read_only: bool = False
    allow_final_bake: bool = True
    approval_risks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonDict
    handler: Callable[[JsonDict], Any]
    mutating: bool = False
    risk: str = "safe"

    def as_function_schema(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class AgentSession:
    """Stateful, toolkit-independent agent interface for Fictional Terrain Studio.

    The session wraps the same :class:`EditorController` used by the desktop UI.
    Tools intentionally return JSON-serializable structures so the same registry
    can back REST, MCP, Ollama/Qwen function calling, tests, or future clients.
    """

    def __init__(
        self,
        controller: EditorController | None = None,
        *,
        policy: AgentPolicy | None = None,
        approval_hook: Callable[[ToolSpec, JsonDict], bool] | None = None,
        event_hook: Callable[[JsonDict], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        self.controller = controller or EditorController()
        self.policy = policy or AgentPolicy()
        self.approval_hook = approval_hook
        self.event_hook = event_hook
        self.cancel_check = cancel_check
        self._lock = RLock()
        self._tools: dict[str, ToolSpec] = {}
        self._events: list[JsonDict] = []
        self._event_seq = 0
        self._register_tools()

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def tool_specs(self) -> list[JsonDict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "mutating": t.mutating,
                "risk": t.risk,
            }
            for t in self.tools
        ]

    def ollama_tools(self) -> list[JsonDict]:
        return [t.as_function_schema() for t in self.tools]

    def execute(self, name: str, arguments: JsonDict | None = None) -> Any:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown FTS tool: {name}") from exc
        args = dict(arguments or {})
        if tool.mutating and self.policy.read_only:
            raise PermissionError(f"Agent session is read-only; tool {name} is not allowed")
        if name == "terrain_bake_final" and not self.policy.allow_final_bake:
            raise PermissionError("Final bake is disabled by the agent-session policy")
        started = perf_counter()
        if tool.risk in set(self.policy.approval_risks):
            approved = bool(self.approval_hook(tool, args)) if self.approval_hook is not None else False
            if not approved:
                self._log_event(
                    tool, args, success=False, duration_ms=(perf_counter()-started)*1000.0,
                    error="ApprovalDenied: user declined operation",
                )
                raise PermissionError(f"User approval required for {name} and was not granted")
        try:
            with self._lock:
                result = tool.handler(args)
        except Exception as exc:
            self._log_event(tool, args, success=False, duration_ms=(perf_counter()-started)*1000.0, error=f"{type(exc).__name__}: {exc}")
            raise
        self._log_event(tool, args, success=True, duration_ms=(perf_counter()-started)*1000.0)
        return _jsonable(result)

    def _log_event(self, tool: ToolSpec, arguments: JsonDict, *, success: bool, duration_ms: float, error: str | None = None) -> None:
        self._event_seq += 1
        self._events.append({
            "seq": self._event_seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool.name,
            "mutating": tool.mutating,
            "risk": tool.risk,
            "success": success,
            "duration_ms": round(float(duration_ms), 3),
            "arguments": _jsonable(arguments),
            "error": error,
        })
        if len(self._events) > 500:
            del self._events[:-500]
        if self.event_hook is not None:
            try:
                self.event_hook(dict(self._events[-1]))
            except Exception:
                pass

    def events(self, limit: int = 100) -> list[JsonDict]:
        n = max(1, min(int(limit), 500))
        return [dict(x) for x in self._events[-n:]]

    def open_initial_project(self, path: str | Path | None) -> None:
        if path:
            self.controller.open_project(path)

    def status(self) -> JsonDict:
        c = self.controller
        project = c.project
        return {
            "ok": True,
            "project_open": c.has_project,
            "project_path": str(c.project_path) if c.project_path else None,
            "project_name": project.name if project else None,
            "object_count": len(project.objects) if project else 0,
            "selected_object_id": c.selected_object_id,
            "selected_point_index": c.selected_point_index,
            "history_depth": len(project.history) if project else 0,
            "redo_depth": len(project.redo_stack) if project else 0,
            "last_report": c.last_report,
            "dirty": c.is_dirty,
            "autosave_path": str(c.autosave_path()) if c.autosave_path() else None,
            "policy": {
                "read_only": self.policy.read_only,
                "allow_final_bake": self.policy.allow_final_bake,
                "approval_risks": list(self.policy.approval_risks),
            },
            "event_count": len(self._events),
        }

    def editor_context(self, view_mode: str = "hillshade") -> JsonDict:
        c = self.controller
        project = c.project
        selected = c.selected_object() if project is not None else None
        report = c.last_report or {}
        hydro = report.get("hydrology") or {}
        preview_history_depth = report.get("history_depth")
        current_history_depth = len(project.history) if project else 0
        context: JsonDict = {
            "view_mode": view_mode,
            "project_open": c.has_project,
            "project_name": project.name if project else None,
            "project_path": str(c.project_path) if c.project_path else None,
            "object_count": len(project.objects) if project else 0,
            "selected_object_id": c.selected_object_id,
            "selected_point_index": c.selected_point_index,
            "preview_ready": c.last_render is not None,
            "preview_stale": bool(c.last_render is not None and preview_history_depth is not None and preview_history_depth != current_history_depth),
            "last_dirty_bounds": list(project.last_dirty_bounds()) if project and project.last_dirty_bounds() else None,
            "project_dirty": c.is_dirty,
        }
        if selected is not None:
            pts = selected.control_points
            bounds = None
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bounds = [min(xs), min(ys), max(xs), max(ys)]
            context["selected_object"] = {
                "id": selected.id,
                "name": selected.name,
                "object_type": selected.object_type,
                "enabled": selected.enabled,
                "control_points": [list(p) for p in pts],
                "control_point_count": len(pts),
                "normalized_bounds": bounds,
                "properties": dict(selected.properties),
            }
        if report:
            context["qa"] = {
                "total_relief_m": report.get("total_relief_m"),
                "slope_mean_percent": report.get("slope_mean_percent"),
                "slope_p95_percent": report.get("slope_p95_percent"),
                "fraction_under_10pct": report.get("fraction_under_10pct"),
                "hydrology": {
                    "conditioned_interior_sinks": hydro.get("conditioned_interior_sinks"),
                    "fill_max_m": hydro.get("fill_max_m"),
                    "significant_depression_area_fraction": hydro.get("significant_depression_area_fraction"),
                    "largest_boundary_accumulation_km2": hydro.get("largest_boundary_accumulation_km2"),
                    "semantic_river_connected": hydro.get("semantic_river_connected"),
                    "semantic_river_valid_outlet": hydro.get("semantic_river_valid_outlet"),
                } if hydro else None,
            }
        if project is not None:
            try:
                cfg = project.compile_config("preview")
                context["bible_profile"] = validate_bible_profile(cfg, project.objects)
            except Exception:
                pass
        return context

    def _add(
        self,
        name: str,
        description: str,
        parameters: JsonDict,
        handler: Callable[[JsonDict], Any],
        *,
        mutating: bool = False,
        risk: str | None = None,
    ) -> None:
        resolved_risk = risk or ("edit" if mutating else "safe")
        self._tools[name] = ToolSpec(name, description, parameters, handler, mutating, resolved_risk)

    def _register_tools(self) -> None:
        obj = {"type": "object", "properties": {}, "additionalProperties": False}
        self._add("session_status", "Return the current Fictional Terrain Studio agent/session status.", obj, lambda _: self.status())
        self._add(
            "session_events",
            "Return recent agent/API tool events for auditing and debugging.",
            _schema({"limit": _integer("Maximum recent events", minimum=1, maximum=500, default=100)}),
            lambda a: {"events": self.events(int(a.get("limit", 100)))},
        )
        self._add(
            "session_editor_context",
            "Return the current editor selection, preview state, and compact QA context for agent reasoning.",
            _schema({"view": _string("Current editor view mode", default="hillshade")}),
            lambda a: self.editor_context(str(a.get("view", "hillshade"))),
        )
        self._add(
            "project_open",
            "Open an existing .ftsproject.json project and make it the active project.",
            _schema({"path": _string("Absolute or relative project path")}, required=["path"]),
            lambda a: self._project_open(a["path"]),
            mutating=True,
            risk="destructive",
        )
        self._add(
            "project_save",
            "Save the active project. Omit path to overwrite the current project file.",
            _schema({"path": _nullable_string("Optional destination .ftsproject.json path")}),
            lambda a: {"path": str(self.controller.save_project(a.get("path")))},
            mutating=True,
            risk="destructive",
        )
        self._add("project_summary", "Return project metadata, object counts, render settings, and history state.", obj, lambda _: self._project_summary())
        self._add("project_list_objects", "List all editable terrain objects in the active project.", obj, lambda _: self._list_objects())
        self._add(
            "project_get_object",
            "Return one editable terrain object by id.",
            _schema({"object_id": _string("Object id")}, required=["object_id"]),
            lambda a: self._require_project().get_object(a["object_id"]).to_dict(),
        )
        self._add(
            "project_select_object",
            "Select an editable object by id.",
            _schema({"object_id": _string("Object id")}, required=["object_id"]),
            lambda a: self._select_object(a["object_id"]),
        )
        self._add(
            "project_add_object",
            "Add a template river, valley, bluff, moraine, lake, coast, harbor, or water-source object and return its id.",
            _schema({"kind": {"type": "string", "enum": ["river", "valley", "bluff", "moraine", "lake", "coast", "harbor", "water_source"]}}, required=["kind"]),
            lambda a: {"object_id": self.controller.add_template_object(a["kind"])},
            mutating=True,
        )
        self._add(
            "project_delete_object",
            "Delete an editable object by id.",
            _schema({"object_id": _string("Object id")}, required=["object_id"]),
            self._delete_object,
            mutating=True,
            risk="destructive",
        )
        self._add(
            "project_update_object",
            "Update an object's name, enabled state, or terrain properties. Properties are merged unless replace_properties is true.",
            _schema(
                {
                    "object_id": _string("Object id"),
                    "name": _nullable_string("New display name"),
                    "enabled": {"type": ["boolean", "null"]},
                    "properties": {"type": ["object", "null"], "additionalProperties": True},
                    "replace_properties": {"type": "boolean", "default": False},
                },
                required=["object_id"],
            ),
            self._update_object,
            mutating=True,
        )
        self._add(
            "spline_move_point",
            "Move an existing spline control point using normalized map coordinates from -1 to 1.",
            _schema(
                {
                    "object_id": _string("Object id"),
                    "point_index": _integer("Zero-based control-point index", minimum=0),
                    "x": _number("Normalized x coordinate, usually -1 to 1"),
                    "y": _number("Normalized y coordinate, usually -1 to 1"),
                },
                required=["object_id", "point_index", "x", "y"],
            ),
            self._move_point,
            mutating=True,
        )
        self._add(
            "spline_insert_point",
            "Insert a control point before the given spline index.",
            _schema(
                {
                    "object_id": _string("Object id"),
                    "point_index": _integer("Insertion index", minimum=0),
                    "x": _number("Normalized x coordinate"),
                    "y": _number("Normalized y coordinate"),
                },
                required=["object_id", "point_index", "x", "y"],
            ),
            self._insert_point,
            mutating=True,
        )
        self._add(
            "spline_delete_point",
            "Delete a spline control point.",
            _schema({"object_id": _string("Object id"), "point_index": _integer("Control-point index", minimum=0)}, required=["object_id", "point_index"]),
            self._delete_point,
            mutating=True,
            risk="destructive",
        )
        self._add("history_undo", "Undo the last project edit.", obj, lambda _: {"changed": self.controller.undo()}, mutating=True)
        self._add("history_redo", "Redo the last undone project edit.", obj, lambda _: {"changed": self.controller.redo()}, mutating=True)
        self._add(
            "terrain_render_preview",
            "Render the editable project at interactive preview resolution and return terrain/hydrology QA metrics.",
            _schema({"hydrology": {"type": "boolean", "default": True}}),
            lambda a: self._render_preview(bool(a.get("hydrology", True))),
        )
        self._add(
            "terrain_save_preview",
            "Render or reuse the preview and save a PNG visualization (hillshade, elevation, slope, buildability, or flow).",
            _schema(
                {
                    "view": {"type": "string", "enum": ["hillshade", "elevation", "slope", "buildability", "flow", "water", "bathymetry"], "default": "hillshade"},
                    "path": _nullable_string("Optional PNG destination. A temporary file is used if omitted."),
                    "max_size": _integer("Preview image maximum dimension", minimum=256, maximum=1200, default=1000),
                }
            ),
            self._save_preview,
        )
        self._add(
            "terrain_bake_final",
            "Bake the active project to a final CS2 4096x4096 16-bit heightmap and full QA output directory.",
            _schema({"out_dir": _string("Destination output directory")}, required=["out_dir"]),
            lambda a: self._bake_final(a["out_dir"]),
            risk="expensive",
        )
        self._add(
            "cs2_workflow_plan",
            "Inspect CS2 vertical encoding, World Map settings, bathymetry state, and proposed documented water-source classes without a final bake.",
            obj,
            lambda _: self._cs2_workflow_plan(),
        )
        self._add(
            "cs2_editor_assistance",
            "Return an ordered CS2 Map Editor checklist plus normalized and meter-offset water-source placements for the current project.",
            obj,
            lambda _: self._cs2_editor_assistance(),
        )
        self._add(
            "gis_export_package",
            "Export the current rendered terrain as a canonical Float32 GeoTIFF plus SOURCE_LOCK semantic-object GeoJSON and GIS manifest for QGIS.",
            _schema({"out_dir": _string("Destination GIS interchange directory")}, required=["out_dir"]),
            lambda a: self.controller.export_gis_package(a["out_dir"]),
            risk="expensive",
        )
        self._add(
            "gis_import_objects",
            "Import FTS/QGIS semantic GeoJSON. Existing hydrography is protected unless allow_hydro_updates is explicitly true.",
            _schema({
                "path": _string("GeoJSON path"),
                "update_existing": {"type": "boolean", "default": False},
                "allow_hydro_updates": {"type": "boolean", "default": False},
            }, required=["path"]),
            lambda a: self.controller.import_gis_objects(
                a["path"], update_existing=bool(a.get("update_existing", False)),
                allow_hydro_updates=bool(a.get("allow_hydro_updates", False)),
            ),
            mutating=True,
            risk="destructive",
        )
        self._add(
            "gis_set_base_dem",
            "Use a georeferenced GeoTIFF as the project SOURCE_LOCK base terrain after validating it against the canonical FTS grid.",
            _schema({
                "path": _string("GeoTIFF path"),
                "resampling": {"type": "string", "enum": ["nearest", "bilinear", "cubic"], "default": "bilinear"},
            }, required=["path"]),
            lambda a: self.controller.set_gis_base_dem(a["path"], resampling=str(a.get("resampling", "bilinear"))),
            mutating=True,
            risk="destructive",
        )
        self._add(
            "cs2_install_heightmaps",
            "Copy a completed FTS production heightmap/worldmap pair into a Cities: Skylines II Heightmaps folder. Uses the official LocalLow path when target_dir is omitted on Windows.",
            _schema({
                "production_dir": _string("Completed FTS production output directory"),
                "target_dir": _nullable_string("Optional Heightmaps directory"),
                "prefix": _nullable_string("Optional destination filename prefix"),
                "overwrite": {"type": "boolean", "default": False},
            }, required=["production_dir"]),
            lambda a: self.controller.install_heightmaps(
                a["production_dir"], a.get("target_dir"), prefix=a.get("prefix"), overwrite=bool(a.get("overwrite", False))
            ),
            mutating=True,
            risk="destructive",
        )
        self._add(
            "landform_search",
            "Search a portable semantic landform library by type, name, tags, and relief.",
            _schema(
                {
                    "library": _string("Path to landform_library.json or its directory"),
                    "kind": _string("Optional semantic type: valley, ridge, terrace, bluff", default=""),
                    "name_contains": _string("Optional name substring", default=""),
                    "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                    "min_relief_m": _number("Minimum p95 absolute relief", default=0.0),
                    "max_relief_m": _number("Maximum p95 absolute relief", default=1.0e9),
                    "limit": _integer("Maximum matches", minimum=1, maximum=50, default=10),
                },
                required=["library"],
            ),
            self._landform_search,
        )

    def _project_open(self, path: str) -> JsonDict:
        autosave = None
        if self.controller.has_project and self.controller.is_dirty:
            saved = self.controller.write_autosave()
            autosave = str(saved) if saved is not None else None
        p = self.controller.open_project(path)
        return {"name": p.name, "path": str(self.controller.project_path), "object_count": len(p.objects), "previous_project_autosave": autosave}

    def _require_project(self):
        if self.controller.project is None:
            raise RuntimeError("No project is open")
        return self.controller.project

    def _project_summary(self) -> JsonDict:
        p = self._require_project()
        rs = p.render_settings
        return {
            "name": p.name,
            "schema_version": p.schema_version,
            "project_path": str(self.controller.project_path) if self.controller.project_path else None,
            "object_count": len(p.objects),
            "enabled_object_count": sum(1 for o in p.objects if o.enabled),
            "history_depth": len(p.history),
            "redo_depth": len(p.redo_stack),
            "last_dirty_bounds": list(p.last_dirty_bounds()) if p.last_dirty_bounds() else None,
            "render_settings": {
                "preview_work_resolution": rs.preview_work_resolution,
                "preview_export_resolution": rs.preview_export_resolution,
                "final_work_resolution": rs.final_work_resolution,
                "final_export_resolution": rs.final_export_resolution,
            },
        }

    def _list_objects(self) -> JsonDict:
        p = self._require_project()
        return {
            "objects": [
                {
                    "id": o.id,
                    "object_type": o.object_type,
                    "name": o.name,
                    "enabled": o.enabled,
                    "control_point_count": len(o.control_points),
                    "kind": o.properties.get("kind"),
                    "role": o.properties.get("role"),
                }
                for o in p.objects
            ]
        }

    def _select_object(self, object_id: str) -> JsonDict:
        obj = self.controller.select_object(object_id)
        return obj.to_dict() if obj else {"selected": None}

    def _delete_object(self, a: JsonDict) -> JsonDict:
        self.controller.select_object(a["object_id"])
        deleted = self.controller.delete_selected_object()
        return {"deleted": deleted, "object_id": a["object_id"]}

    def _update_object(self, a: JsonDict) -> JsonDict:
        self.controller.select_object(a["object_id"])
        self.controller.update_selected_object(
            name=a.get("name"),
            enabled=a.get("enabled"),
            properties=a.get("properties"),
            replace_properties=bool(a.get("replace_properties", False)),
        )
        obj = self.controller.selected_object()
        return obj.to_dict() if obj else {"object_id": a["object_id"]}

    def _move_point(self, a: JsonDict) -> JsonDict:
        self.controller.move_point(a["object_id"], int(a["point_index"]), float(a["x"]), float(a["y"]))
        return self._point_result(a["object_id"], int(a["point_index"]))

    def _insert_point(self, a: JsonDict) -> JsonDict:
        self.controller.insert_point(a["object_id"], int(a["point_index"]), float(a["x"]), float(a["y"]))
        return self._point_result(a["object_id"], int(a["point_index"]))

    def _delete_point(self, a: JsonDict) -> JsonDict:
        self.controller.delete_point(a["object_id"], int(a["point_index"]))
        obj = self._require_project().get_object(a["object_id"])
        return {"object_id": obj.id, "control_points": [list(p) for p in obj.control_points]}

    def _point_result(self, object_id: str, point_index: int) -> JsonDict:
        obj = self._require_project().get_object(object_id)
        return {
            "object_id": object_id,
            "point_index": point_index,
            "point": list(obj.control_points[point_index]),
            "dirty_bounds": list(self._require_project().last_dirty_bounds()) if self._require_project().last_dirty_bounds() else None,
        }

    def _render_preview(self, hydrology: bool) -> JsonDict:
        result = self.controller.render_preview(hydrology=hydrology, cancel_check=self.cancel_check)
        report = dict(self.controller.last_report or {})
        report.update({"work_resolution": result.config.work_resolution, "export_resolution": result.config.export_resolution})
        return report

    def _save_preview(self, a: JsonDict) -> JsonDict:
        view = str(a.get("view", "hillshade"))
        preview = self.controller.preview_image(view=view, max_size=int(a.get("max_size", 1000)))
        path_arg = a.get("path")
        if path_arg:
            out = Path(path_arg).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            fd, tmp = tempfile.mkstemp(prefix="fts-preview-", suffix=f"-{preview.mode}.png")
            import os
            os.close(fd)
            out = Path(tmp)
        preview.image.save(out)
        return {"path": str(out), "view": preview.mode, "size": list(preview.image.size), "report": preview.report}

    def _bake_final(self, out_dir: str) -> JsonDict:
        report = self.controller.bake_final(out_dir, cancel_check=self.cancel_check)
        return {"out_dir": str(Path(out_dir).expanduser().resolve()), "report": report}

    def _cs2_workflow_plan(self) -> JsonDict:
        project = self._require_project()
        if self.controller.last_render is None:
            self.controller.render_preview(hydrology=False, cancel_check=self.cancel_check)
        result = self.controller.last_render
        assert result is not None
        cfg = result.config
        water = np.asarray(result.metadata.get("water_mask"), dtype=bool)
        prod_elev, prod_water, water_metrics = apply_water_terrain(result.elevation_m, water, cfg, objects=project.objects)
        return {
            "encoding": vertical_calibration(prod_elev, cfg),
            "water_sources": plan_water_sources(cfg, prod_elev, prod_water, objects=project.objects),
            "semantic_water": water_metrics,
            "world_map_enabled": bool(cfg.production.world_map_enabled),
            "world_extent_km": float(cfg.production.nominal_world_extent_km),
            "bathymetry_enabled": bool(cfg.production.water_enabled),
            "bathymetry_applied": bool(cfg.production.apply_bathymetry),
            "littoral_width_m": float(cfg.production.littoral_width_m),
            "note": "Final 4096x4096 format and exact World Map center correspondence are validated during terrain_bake_final.",
        }


    def _cs2_editor_assistance(self) -> JsonDict:
        project = self._require_project()
        if self.controller.last_render is None:
            self.controller.render_preview(hydrology=False, cancel_check=self.cancel_check)
        result = self.controller.last_render
        assert result is not None
        cfg = result.config
        base_water = np.asarray(result.metadata.get("water_mask"), dtype=bool)
        prod_elev, prod_water, _ = apply_water_terrain(result.elevation_m, base_water, cfg, objects=project.objects)
        sources = plan_water_sources(cfg, prod_elev, prod_water, objects=project.objects)
        return build_editor_assistance(cfg, sources, semantic_objects=project.objects)

    def _landform_search(self, a: JsonDict) -> JsonDict:
        entries = load_landform_index(a["library"])
        matches = search_landforms(
            entries,
            kind=str(a.get("kind", "")),
            name_contains=str(a.get("name_contains", "")),
            tags=list(a.get("tags") or []),
            min_relief_m=float(a.get("min_relief_m", 0.0)),
            max_relief_m=float(a.get("max_relief_m", 1.0e9)),
            limit=int(a.get("limit", 10)),
        )
        return {
            "matches": [
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
                for e in matches
            ]
        }


def _schema(properties: JsonDict, required: list[str] | None = None) -> JsonDict:
    out: JsonDict = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        out["required"] = required
    return out


def _string(description: str, default: str | None = None) -> JsonDict:
    out: JsonDict = {"type": "string", "description": description}
    if default is not None:
        out["default"] = default
    return out


def _nullable_string(description: str) -> JsonDict:
    return {"type": ["string", "null"], "description": description, "default": None}


def _number(description: str, default: float | None = None) -> JsonDict:
    out: JsonDict = {"type": "number", "description": description}
    if default is not None:
        out["default"] = default
    return out


def _integer(description: str, minimum: int | None = None, maximum: int | None = None, default: int | None = None) -> JsonDict:
    out: JsonDict = {"type": "integer", "description": description}
    if minimum is not None:
        out["minimum"] = minimum
    if maximum is not None:
        out["maximum"] = maximum
    if default is not None:
        out["default"] = default
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
