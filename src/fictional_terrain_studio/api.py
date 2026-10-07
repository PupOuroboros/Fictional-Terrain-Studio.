from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import argparse
import secrets

from .agent_tools import AgentPolicy, AgentSession


def create_app(*, project: str | Path | None = None, api_token: str | None = None, read_only: bool = False, allow_final_bake: bool = True):
    try:
        from fastapi import FastAPI, Header, HTTPException, Query
        from fastapi.responses import Response
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError("FastAPI is required for the local agent API. Install fictional-terrain-studio[agent].") from exc

    session = AgentSession(policy=AgentPolicy(read_only=read_only, allow_final_bake=allow_final_bake))
    session.open_initial_project(project)
    app = FastAPI(
        title="Fictional Terrain Studio Agent API",
        version="0.11.0",
        description="Local API over the same authoring controller used by the desktop editor.",
    )

    def authorize(authorization: str | None) -> None:
        if not api_token:
            return
        expected = f"Bearer {api_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid API token")

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return session.status()

    @app.get("/tools")
    def tools(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"tools": session.tool_specs()}

    @app.get("/ollama-tools")
    def ollama_tools(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"tools": session.ollama_tools()}

    @app.post("/tool/{tool_name}")
    def call_tool(tool_name: str, call: dict[str, Any], authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return {"ok": True, "result": session.execute(tool_name, dict(call.get("arguments") or {}))}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


    @app.get("/events")
    def events(limit: int = Query(default=100, ge=1, le=500), authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"events": session.events(limit)}

    @app.get("/project")
    def project_summary(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return session.execute("project_summary", {})

    @app.get("/context")
    def editor_context(
        view: str = Query(default="hillshade", pattern="^(hillshade|elevation|slope|buildability|flow)$"),
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        return session.execute("session_editor_context", {"view": view})

    @app.get("/qa")
    def qa(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return session.controller.last_report or {}

    @app.get("/preview.png")
    def preview_png(
        view: str = Query(default="hillshade", pattern="^(hillshade|elevation|slope|buildability|flow)$"),
        max_size: int = Query(default=1000, ge=256, le=1200),
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        p = session.controller.preview_image(view=view, max_size=max_size)
        buf = BytesIO()
        p.image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png", headers={"X-FTS-View": p.mode})

    app.state.fts_session = session
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fts-api", description="Run the local Fictional Terrain Studio agent API")
    parser.add_argument("--project", default=None, help="Optional .ftsproject.json to open at startup")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; defaults to localhost only")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=None, help="Optional bearer token")
    parser.add_argument("--read-only", action="store_true", help="Reject all mutating project tools")
    parser.add_argument("--no-final-bake", action="store_true", help="Disable final bake through the API")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required. Install fictional-terrain-studio[agent].") from exc
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.token:
        parser.error("Binding beyond localhost requires --token")
    app = create_app(project=args.project, api_token=args.token, read_only=args.read_only, allow_final_bake=not args.no_final_bake)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
