"""Main FastAPI Application Entrypoint for EcoSort AI."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .core.config import settings
from .api.router import api_router

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "AI-Powered Multilingual Multimodal Standalone Waste Segregation Kiosk Backend "
        "for Bharat AI-SoC Challenge 2026–27 (Problem Statement 2: Smart Recycling & Waste Segregation). "
        "Strictly grounded in the official Solid Waste Management Rules, 2026 (Notification S.O. 388(E))."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration for React + Vite development and kiosk display
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "*"  # Allow local network kiosk display devices
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount central API router under /api
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", tags=["Root"])
def root():
    """Root metadata endpoint."""
    return {
        "app": settings.app_name,
        "version": settings.version,
        "challenge": "Bharat AI-SoC Challenge 2026–27",
        "problem_statement": "Problem Statement 2: Smart Recycling and Waste Segregation",
        "target_hardware": settings.target_hardware,
        "statutory_primary_source": "Solid Waste Management Rules, 2026 (Notification S.O. 388(E))",
        "docs_url": "/docs",
        "api_prefix": settings.api_prefix,
    }


@app.get("/health", tags=["Root"])
def direct_health():
    """Direct root health check (convenience redirect/alias to /api/health)."""
    from .api.routes.health import get_health
    return get_health()


# Global exception handler for standardized error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
