"""
Optimization API Endpoints

This module provides REST endpoints for the canonical optimization functionality.
The web interface uses these endpoints to run NSGA-II optimization and retrieve
Pareto optimal parameter sets.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
import traceback
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import uuid

# Add project paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from core.canonical_optimization import (
    CanonicalOptimizer,
    OptimizationBounds,
    OptimizationStrategy as CoreOptimizationStrategy,
    optimize_for_strategy,
    find_pareto_optimal_parameters,
    validate_parameter_set
)
from core.canonical_fee_mechanism import VaultInitMode as CoreVaultInitMode
from api.models import (
    OptimizationRequest,
    OptimizationResponse,
    IndividualSolution,
    OptimizationStrategy,
    VaultInitMode,
    validate_l1_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/optimization", tags=["optimization"])

# Global state for background optimization jobs
_optimization_jobs: Dict[str, Dict[str, Any]] = {}
_job_executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent optimizations


def _convert_optimization_strategy(strategy: OptimizationStrategy) -> CoreOptimizationStrategy:
    """Convert API enum to core enum."""
    mapping = {
        OptimizationStrategy.user_centric: CoreOptimizationStrategy.USER_CENTRIC,
        OptimizationStrategy.protocol_centric: CoreOptimizationStrategy.PROTOCOL_CENTRIC,
        OptimizationStrategy.balanced: CoreOptimizationStrategy.BALANCED,
        OptimizationStrategy.crisis_resilient: CoreOptimizationStrategy.CRISIS_RESILIENT,
        OptimizationStrategy.capital_efficient: CoreOptimizationStrategy.CAPITAL_EFFICIENT
    }
    return mapping[strategy]


def _convert_vault_init_mode(mode: VaultInitMode) -> CoreVaultInitMode:
    """Convert API enum to core enum."""
    mapping = {
        VaultInitMode.target: CoreVaultInitMode.TARGET,
        VaultInitMode.deficit: CoreVaultInitMode.DEFICIT,
        VaultInitMode.surplus: CoreVaultInitMode.SURPLUS,
        VaultInitMode.custom: CoreVaultInitMode.CUSTOM
    }
    return mapping[mode]


def _run_optimization_job(job_id: str, request: OptimizationRequest) -> None:
    """Run optimization in background thread."""
    try:
        logger.info(f"Starting optimization job {job_id}")
        start_time = time.time()

        # Update job status
        _optimization_jobs[job_id]["status"] = "running"
        _optimization_jobs[job_id]["start_time"] = start_time

        # Validate L1 data if provided
        l1_data = None
        if request.l1_data:
            validate_l1_data(request.l1_data)
            l1_data = request.l1_data

        # Create optimizer with custom bounds if provided
        bounds = OptimizationBounds()
        if request.mu_min is not None:
            bounds.mu_min = request.mu_min
        if request.mu_max is not None:
            bounds.mu_max = request.mu_max
        if request.nu_min is not None:
            bounds.nu_min = request.nu_min
        if request.nu_max is not None:
            bounds.nu_max = request.nu_max
        if request.H_min is not None:
            bounds.H_min = request.H_min
        if request.H_max is not None:
            bounds.H_max = request.H_max

        optimizer = CanonicalOptimizer(bounds)
        optimizer.simulation_steps = request.simulation_steps

        # Convert enums
        strategy = _convert_optimization_strategy(request.strategy)
        vault_init = _convert_vault_init_mode(request.vault_init)

        # Progress callback
        def progress_callback(generation: int, total_generations: int, population: List[Any]):
            progress = generation / total_generations
            _optimization_jobs[job_id]["progress"] = progress
            _optimization_jobs[job_id]["generation"] = generation
            _optimization_jobs[job_id]["total_generations"] = total_generations
            logger.info(f"Job {job_id}: Generation {generation}/{total_generations} ({progress*100:.1f}%)")

        # Run optimization
        results = optimizer.optimize(
            strategy=strategy,
            population_size=request.population_size,
            generations=request.generations,
            l1_data=l1_data,
            vault_init=vault_init,
            progress_callback=progress_callback
        )

        # Convert results to API format
        pareto_solutions = []
        for individual in results['pareto_front']:
            if individual.objectives is None:
                continue

            # Calculate derived metrics (simplified from objectives)
            user_score = max(0.0, 1.0 + individual.objectives[0])  # Reverse negative
            safety_score = max(0.0, 1.0 + individual.objectives[4])
            efficiency_score = max(0.0, 1.0 + individual.objectives[8])
            overall_score = (user_score + safety_score + efficiency_score) / 3.0

            solution = IndividualSolution(
                mu=individual.mu,
                nu=individual.nu,
                H=individual.H,
                user_experience_score=user_score,
                protocol_safety_score=safety_score,
                economic_efficiency_score=efficiency_score,
                overall_score=overall_score,
                average_fee_gwei=10.0 * (1.0 - user_score),  # Approximate inverse
                fee_stability_cv=max(0.1, 1.0 - user_score),
                time_underfunded_pct=max(0.0, 20.0 * (1.0 - safety_score)),
                l1_tracking_error=max(0.1, 1.0 - efficiency_score),
                pareto_rank=individual.rank,
                crowding_distance=individual.crowding_distance,
                simulation_time=individual.simulation_time
            )
            pareto_solutions.append(solution)

        # Calculate convergence history for API
        convergence_history = []
        for entry in results.get('convergence_history', []):
            api_entry = {
                "generation": entry.get('generation', 0),
                "pareto_size": entry.get('pareto_size', 0),
                "hypervolume": entry.get('hypervolume', 0.0),
                "spread": entry.get('spread', 0.0)
            }
            convergence_history.append(api_entry)

        # Create response
        optimization_response = OptimizationResponse(
            strategy=request.strategy,
            pareto_solutions=pareto_solutions,
            total_generations=request.generations,
            total_evaluations=request.population_size * request.generations,
            optimization_time=time.time() - start_time,
            hypervolume=results.get('hypervolume', 0.0),
            spread=results.get('spread', 0.0),
            n_pareto_solutions=len(pareto_solutions),
            convergence_history=convergence_history
        )

        # Update job status
        _optimization_jobs[job_id]["status"] = "completed"
        _optimization_jobs[job_id]["result"] = optimization_response
        _optimization_jobs[job_id]["end_time"] = time.time()

        logger.info(f"Optimization job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Optimization job {job_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update job status
        _optimization_jobs[job_id]["status"] = "failed"
        _optimization_jobs[job_id]["error"] = str(e)
        _optimization_jobs[job_id]["end_time"] = time.time()


@router.post("/start", response_model=Dict[str, str])
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Start a new optimization job in the background.

    This endpoint initiates a long-running optimization process and returns
    a job ID that can be used to check progress and retrieve results.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Initialize job tracking
        _optimization_jobs[job_id] = {
            "status": "starting",
            "request": request,
            "progress": 0.0,
            "generation": 0,
            "total_generations": request.generations,
            "created_time": time.time()
        }

        # Submit job to thread pool
        future = _job_executor.submit(_run_optimization_job, job_id, request)
        _optimization_jobs[job_id]["future"] = future

        logger.info(f"Started optimization job {job_id}")

        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Optimization job started with {request.population_size} individuals, {request.generations} generations"
        }

    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to start optimization: {str(e)}")


@router.get("/status/{job_id}")
async def get_optimization_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a running optimization job.

    Returns progress information and current status for the specified job ID.
    """
    if job_id not in _optimization_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _optimization_jobs[job_id]

    status_info = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "generation": job.get("generation", 0),
        "total_generations": job.get("total_generations", 0),
        "created_time": job["created_time"]
    }

    if "start_time" in job:
        status_info["start_time"] = job["start_time"]
        status_info["elapsed_time"] = time.time() - job["start_time"]

    if "end_time" in job:
        status_info["end_time"] = job["end_time"]
        status_info["total_time"] = job["end_time"] - job.get("start_time", job["created_time"])

    if "error" in job:
        status_info["error"] = job["error"]

    return status_info


@router.get("/result/{job_id}", response_model=OptimizationResponse)
async def get_optimization_result(job_id: str) -> OptimizationResponse:
    """
    Get the results of a completed optimization job.

    Returns the complete optimization results including Pareto front solutions
    and convergence analysis.
    """
    if job_id not in _optimization_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _optimization_jobs[job_id]

    if job["status"] == "completed":
        if "result" in job:
            return job["result"]
        else:
            raise HTTPException(status_code=500, detail="Job completed but result not found")
    elif job["status"] == "failed":
        error_msg = job.get("error", "Unknown error")
        raise HTTPException(status_code=400, detail=f"Optimization failed: {error_msg}")
    elif job["status"] in ["starting", "running"]:
        raise HTTPException(status_code=202, detail=f"Job is still {job['status']}")
    else:
        raise HTTPException(status_code=400, detail=f"Job in unexpected status: {job['status']}")


@router.delete("/cancel/{job_id}")
async def cancel_optimization(job_id: str) -> Dict[str, str]:
    """
    Cancel a running optimization job.

    Attempts to stop the optimization process and clean up resources.
    """
    if job_id not in _optimization_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _optimization_jobs[job_id]

    if job["status"] in ["completed", "failed"]:
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Job already finished"
        }

    # Try to cancel the future
    if "future" in job:
        cancelled = job["future"].cancel()
        if cancelled:
            job["status"] = "cancelled"
            job["end_time"] = time.time()
            logger.info(f"Successfully cancelled job {job_id}")
            return {
                "job_id": job_id,
                "status": "cancelled",
                "message": "Job cancelled successfully"
            }
        else:
            logger.warning(f"Could not cancel job {job_id} - may be running")
            return {
                "job_id": job_id,
                "status": job["status"],
                "message": "Could not cancel job - it may be running"
            }

    return {
        "job_id": job_id,
        "status": "unknown",
        "message": "Job state unclear"
    }


@router.post("/quick", response_model=OptimizationResponse)
async def quick_optimization(request: OptimizationRequest) -> OptimizationResponse:
    """
    Run a quick optimization synchronously.

    This endpoint runs a smaller optimization directly and returns results
    immediately. Use for small population sizes and few generations.
    """
    try:
        # Limit parameters for quick optimization
        if request.population_size > 50:
            logger.warning(f"Reducing population size from {request.population_size} to 50 for quick optimization")
            request.population_size = 50

        if request.generations > 20:
            logger.warning(f"Reducing generations from {request.generations} to 20 for quick optimization")
            request.generations = 20

        # Run optimization directly
        start_time = time.time()

        strategy = _convert_optimization_strategy(request.strategy)
        results = optimize_for_strategy(
            strategy=strategy,
            population_size=request.population_size,
            generations=request.generations
        )

        # Convert to API format (simplified)
        pareto_solutions = []
        for individual in results['pareto_front'][:10]:  # Limit to top 10 solutions
            if individual.objectives is None:
                continue

            # Calculate scores (simplified approximation)
            user_score = max(0.0, min(1.0, 0.8 - individual.mu))  # Prefer low mu
            safety_score = max(0.0, min(1.0, individual.nu))      # Prefer higher nu
            efficiency_score = max(0.0, min(1.0, 1.0 - individual.nu * 0.5))  # Balance

            solution = IndividualSolution(
                mu=individual.mu,
                nu=individual.nu,
                H=individual.H,
                user_experience_score=user_score,
                protocol_safety_score=safety_score,
                economic_efficiency_score=efficiency_score,
                overall_score=(user_score + safety_score + efficiency_score) / 3.0,
                average_fee_gwei=5.0 + individual.mu * 10.0,  # Rough approximation
                fee_stability_cv=0.2 + individual.nu * 0.5,
                time_underfunded_pct=max(0.0, 10.0 * (1.0 - individual.nu)),
                l1_tracking_error=0.1 + individual.mu * 0.4,
                pareto_rank=individual.rank,
                crowding_distance=individual.crowding_distance,
                simulation_time=0.1  # Quick estimation
            )
            pareto_solutions.append(solution)

        optimization_time = time.time() - start_time

        return OptimizationResponse(
            strategy=request.strategy,
            pareto_solutions=pareto_solutions,
            total_generations=request.generations,
            total_evaluations=request.population_size * request.generations,
            optimization_time=optimization_time,
            hypervolume=results.get('hypervolume', 0.0),
            spread=results.get('spread', 0.0),
            n_pareto_solutions=len(pareto_solutions),
            convergence_history=[]  # Skip for quick optimization
        )

    except Exception as e:
        logger.error(f"Quick optimization error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Quick optimization failed: {str(e)}")


@router.post("/validate-solution")
async def validate_solution(mu: float, nu: float, H: int, l1_data: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Validate a specific parameter combination and return performance metrics.

    This endpoint evaluates a single parameter set and returns detailed
    performance analysis.
    """
    try:
        # Validate parameters
        if not (0.0 <= mu <= 1.0):
            raise HTTPException(status_code=400, detail=f"mu must be between 0.0 and 1.0, got {mu}")
        if not (0.0 <= nu <= 1.0):
            raise HTTPException(status_code=400, detail=f"nu must be between 0.0 and 1.0, got {nu}")
        if H % 6 != 0:
            raise HTTPException(status_code=400, detail=f"H must be multiple of 6, got {H}")

        # Validate L1 data if provided
        if l1_data:
            validate_l1_data(l1_data)

        # Run validation
        validation_result = validate_parameter_set(mu, nu, H, l1_data)

        return {
            "parameters": {"mu": mu, "nu": nu, "H": H},
            "validation_result": validation_result,
            "message": "Parameter validation completed successfully"
        }

    except Exception as e:
        logger.error(f"Solution validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Solution validation failed: {str(e)}")


@router.get("/jobs")
async def list_optimization_jobs() -> Dict[str, Any]:
    """
    List all optimization jobs and their current status.

    Returns a summary of all optimization jobs, including active and completed ones.
    """
    try:
        jobs_summary = []

        for job_id, job in _optimization_jobs.items():
            summary = {
                "job_id": job_id,
                "status": job["status"],
                "created_time": job["created_time"],
                "strategy": job["request"].strategy if "request" in job else "unknown",
                "population_size": job["request"].population_size if "request" in job else 0,
                "generations": job["request"].generations if "request" in job else 0
            }

            if "start_time" in job:
                summary["start_time"] = job["start_time"]

            if "end_time" in job:
                summary["end_time"] = job["end_time"]
                summary["total_time"] = job["end_time"] - job.get("start_time", job["created_time"])

            if job["status"] == "running":
                summary["progress"] = job.get("progress", 0.0)
                summary["generation"] = job.get("generation", 0)

            if job["status"] == "completed" and "result" in job:
                summary["n_solutions"] = len(job["result"].pareto_solutions)

            jobs_summary.append(summary)

        # Sort by creation time (newest first)
        jobs_summary.sort(key=lambda x: x["created_time"], reverse=True)

        return {
            "total_jobs": len(jobs_summary),
            "active_jobs": len([j for j in jobs_summary if j["status"] in ["starting", "running"]]),
            "completed_jobs": len([j for j in jobs_summary if j["status"] == "completed"]),
            "jobs": jobs_summary
        }

    except Exception as e:
        logger.error(f"Jobs listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.delete("/cleanup")
async def cleanup_old_jobs() -> Dict[str, str]:
    """
    Clean up old completed and failed optimization jobs.

    Removes job data for completed/failed jobs older than 1 hour to free memory.
    """
    try:
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour

        jobs_to_remove = []
        for job_id, job in _optimization_jobs.items():
            if job["status"] in ["completed", "failed", "cancelled"]:
                job_age = current_time - job.get("end_time", job["created_time"])
                if job_age > cleanup_threshold:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del _optimization_jobs[job_id]

        logger.info(f"Cleaned up {len(jobs_to_remove)} old optimization jobs")

        return {
            "message": f"Cleaned up {len(jobs_to_remove)} old jobs",
            "remaining_jobs": len(_optimization_jobs)
        }

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# Utility endpoints

@router.get("/health")
async def health_check():
    """Health check for optimization service."""
    try:
        # Test basic optimization functionality
        bounds = OptimizationBounds()
        optimizer = CanonicalOptimizer(bounds)

        return {
            "status": "healthy",
            "optimization_service": "operational",
            "active_jobs": len([j for j in _optimization_jobs.values() if j["status"] in ["starting", "running"]]),
            "total_jobs": len(_optimization_jobs)
        }

    except Exception as e:
        logger.error(f"Optimization health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }