"""
Taiko Fee Mechanism API Server

FastAPI server that provides REST endpoints for the canonical fee mechanism
implementations. This server bridges the JavaScript web interface with the
Python canonical implementations, ensuring consistency across all components.

Features:
- Fee calculation and simulation endpoints
- Multi-objective optimization endpoints
- Performance metrics calculation
- Parameter validation and presets
- Real-time progress tracking for long-running operations
- Comprehensive error handling and logging
- CORS support for web interface integration

Usage:
    # Development
    python src/api/server.py

    # Production with uvicorn
    uvicorn src.api.server:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import sys
import os
from typing import Dict, Any
import traceback

# Add project paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import API routers
from api.fee_api import router as fee_router
from api.optimization_api import router as optimization_router
from api.models import ErrorResponse, HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('taiko_api.log') if os.access(os.getcwd(), os.W_OK) else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Taiko Fee Mechanism API",
    description="REST API for Taiko fee mechanism calculations, optimization, and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Server startup time for uptime calculation
_server_start_time = time.time()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Python HTTP server
        "http://localhost:3000",  # Vite dev server
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
        "file://",  # For local file access
        "*"  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allow all hosts in development
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
    logger.error(traceback.format_exc())

    # Don't expose internal errors in production
    if os.getenv("ENVIRONMENT") == "production":
        error_message = "Internal server error"
        details = None
    else:
        error_message = str(exc)
        details = {"traceback": traceback.format_exc()}

    error_response = ErrorResponse(
        error="InternalServerError",
        message=error_message,
        details=details
    )

    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests for debugging."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        logger.info(f"Response: {response.status_code} in {process_time:.3f}s")

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url} in {process_time:.3f}s - {e}")
        raise


# Include API routers
app.include_router(fee_router)
app.include_router(optimization_router)


# Main endpoints
@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "service": "Taiko Fee Mechanism API",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": time.time() - _server_start_time,
        "endpoints": {
            "fee_mechanism": "/api/fee/*",
            "optimization": "/api/optimization/*",
            "health": "/health",
            "docs": "/docs"
        },
        "canonical_modules": {
            "fee_mechanism": "canonical_fee_mechanism.py",
            "optimization": "canonical_optimization.py",
            "metrics": "canonical_metrics.py"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Comprehensive health check endpoint."""
    try:
        uptime = time.time() - _server_start_time

        # Test canonical module imports
        try:
            from core.canonical_fee_mechanism import create_default_calculator
            from core.canonical_optimization import CanonicalOptimizer
            from core.canonical_metrics import CanonicalMetricsCalculator

            # Quick functionality test
            calculator = create_default_calculator()
            test_fee = calculator.calculate_estimated_fee(0.001, 0.0)

            canonical_modules_loaded = True

        except Exception as e:
            logger.error(f"Canonical modules test failed: {e}")
            canonical_modules_loaded = False

        # Calculate memory usage (simplified)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0

        return HealthCheckResponse(
            status="healthy",
            uptime_seconds=uptime,
            canonical_modules_loaded=canonical_modules_loaded,
            memory_usage_mb=memory_mb
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            uptime_seconds=time.time() - _server_start_time,
            canonical_modules_loaded=False,
            memory_usage_mb=0.0
        )


@app.get("/version")
async def get_version() -> Dict[str, str]:
    """Get API version information."""
    return {
        "api_version": "1.0.0",
        "fastapi_version": "0.68.0+",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed server status."""
    try:
        from api.optimization_api import _optimization_jobs

        uptime = time.time() - _server_start_time

        # Count optimization jobs by status
        job_counts = {}
        for job in _optimization_jobs.values():
            status = job["status"]
            job_counts[status] = job_counts.get(status, 0) + 1

        return {
            "server_status": "running",
            "uptime_seconds": uptime,
            "uptime_human": f"{uptime/3600:.1f} hours",
            "optimization_jobs": {
                "total": len(_optimization_jobs),
                "by_status": job_counts
            },
            "memory_info": _get_memory_info(),
            "endpoints_available": [
                "/api/fee/calculate",
                "/api/fee/simulate",
                "/api/fee/metrics",
                "/api/optimization/start",
                "/api/optimization/quick"
            ]
        }

    except Exception as e:
        logger.error(f"Status endpoint failed: {e}")
        return {
            "server_status": "error",
            "error": str(e),
            "uptime_seconds": time.time() - _server_start_time
        }


def _get_memory_info() -> Dict[str, float]:
    """Get memory usage information."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Server startup initialization."""
    logger.info("🚀 Taiko Fee Mechanism API Server starting up...")
    logger.info(f"📍 Server PID: {os.getpid()}")
    logger.info(f"🐍 Python version: {sys.version}")

    try:
        # Test canonical module imports
        from core.canonical_fee_mechanism import get_optimal_parameters
        from core.canonical_optimization import optimize_for_strategy
        from core.canonical_metrics import calculate_basic_metrics

        logger.info("✅ All canonical modules loaded successfully")

        # Test basic functionality
        optimal_params = get_optimal_parameters()
        logger.info(f"✅ Optimal parameters: μ={optimal_params['mu']}, ν={optimal_params['nu']}, H={optimal_params['H']}")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(traceback.format_exc())
        raise

    logger.info("🎉 Taiko Fee Mechanism API Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown cleanup."""
    logger.info("🛑 Taiko Fee Mechanism API Server shutting down...")

    try:
        # Cleanup optimization jobs
        from api.optimization_api import _optimization_jobs, _job_executor

        # Cancel running jobs
        cancelled_jobs = 0
        for job_id, job in _optimization_jobs.items():
            if job["status"] in ["starting", "running"] and "future" in job:
                if job["future"].cancel():
                    cancelled_jobs += 1

        logger.info(f"🧹 Cancelled {cancelled_jobs} running optimization jobs")

        # Shutdown executor
        _job_executor.shutdown(wait=False)
        logger.info("🧹 Optimization executor shutdown")

    except Exception as e:
        logger.error(f"❌ Shutdown cleanup error: {e}")

    logger.info("👋 Taiko Fee Mechanism API Server shutdown complete")


# Development server
def run_dev_server():
    """Run development server with auto-reload."""
    logger.info("🔧 Starting development server...")

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        reload_dirs=[project_root],
        log_level="info",
        access_log=True
    )


# Production server configuration
def run_production_server():
    """Run production server with optimized settings."""
    logger.info("🚀 Starting production server...")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        workers=2,  # Limited workers due to optimization memory usage
        log_level="warning",
        access_log=False
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Taiko Fee Mechanism API Server")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev", help="Server mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")

    args = parser.parse_args()

    if args.mode == "prod":
        run_production_server()
    else:
        run_dev_server()