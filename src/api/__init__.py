"""
Taiko Fee Mechanism API Package

This package provides REST API endpoints for the canonical fee mechanism
implementations, enabling the web interface to access the authoritative
Python implementations.

Modules:
- server: Main FastAPI application server
- fee_api: Fee calculation and simulation endpoints
- optimization_api: Multi-objective optimization endpoints
- models: Pydantic data models for request/response validation

Usage:
    # Start development server
    python -m src.api.server --mode dev

    # Start production server
    python -m src.api.server --mode prod
"""

__version__ = "1.0.0"
__author__ = "Taiko Fee Research Team"

from .models import *
from .fee_api import router as fee_router
from .optimization_api import router as optimization_router