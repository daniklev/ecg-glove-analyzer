from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import json
import traceback
import logging
import uvicorn
import time
from datetime import datetime
from ecg_glove import EcgGlove
from ecg_filters import HPFilterType

app = FastAPI(title="ECG Quality Analysis API", version="1.0.1")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FilterConfig(BaseModel):
    samplingRate: int
    cleanMethod: str
    peakMethod: str
    notchFrequencies: List[float]
    spikeRemoval: bool
    hpFilterType: str
    powerlineFreq: float
    enableBaselineCorrection: bool
    enableSmoothing: bool
    smoothingWindow: int
    useAmbulanceWeights: bool


class EcgQualityRequest(BaseModel):
    rawData: str  # Base64
    filterConfig: FilterConfig


def safe_float_conversion(value, default=0.0):
    """Safely convert a value to float, handling None and non-numeric values."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def ensure_serializable(obj):
    """Ensure all values in the object are JSON serializable."""
    if isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        # Convert numpy types and other non-serializable types to basic Python types
        try:
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            elif hasattr(obj, "tolist"):  # numpy array
                return obj.tolist()
            else:
                return str(obj)
        except Exception:
            return str(obj)


def get_quality_grade(overall_quality: float) -> str:
    """Convert quality score to grade."""
    if overall_quality >= 0.75:
        return "good"
    elif overall_quality >= 0.51:
        return "normal"
    else:
        return "bad"


def extract_problems(lead_quality: Dict[str, Dict[str, Any]]) -> str:
    """Extract all problems from all leads into a single string."""
    all_problems = set()

    for lead_name, lead_data in lead_quality.items():
        if isinstance(lead_data, dict):
            if lead_data.get("Muscle_Artifact", False):
                all_problems.add("muscle movement interference")
            if lead_data.get("Bad_Electrode_Contact", False):
                all_problems.add("poor electrode contact")
            if lead_data.get("Powerline_Interference", False):
                all_problems.add("electrical interference")
            if lead_data.get("Baseline_Drift", False):
                all_problems.add("baseline wandering")
            if lead_data.get("Low_SNR", False):
                all_problems.add("low signal quality")

    return ", ".join(sorted(all_problems)) if all_problems else "none"


@app.post("/check_ecg_quality")
async def check_ecg_quality(req: EcgQualityRequest):
    """
    Analyze ECG signal quality using the actual ECG processing pipeline.
    """
    start_time = time.time()
    logger.info("Starting ECG quality analysis")

    try:
        # Decode the raw ECG data
        decode_start = time.time()
        raw_bytes = base64.b64decode(req.rawData)
        decode_time = time.time() - decode_start
        logger.info(f"Data decoding completed in {decode_time:.3f}s")

        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Raw ECG data is empty")

        # Convert string to enum for HP filter type
        filter_setup_start = time.time()
        try:
            hp_filter_type = getattr(
                HPFilterType, req.filterConfig.hpFilterType, HPFilterType.HP015
            )
        except AttributeError:
            hp_filter_type = HPFilterType.HP015

        # Initialize ECG processor with the provided configuration
        ecg_glove = EcgGlove(
            sampling_rate=req.filterConfig.samplingRate,
            clean_method=req.filterConfig.cleanMethod,
            peak_method=req.filterConfig.peakMethod,
            filters=[int(round(freq)) for freq in req.filterConfig.notchFrequencies],
            spike_removal=req.filterConfig.spikeRemoval,
            hp_filter_type=hp_filter_type,
            powerline_freq=int(round(req.filterConfig.powerlineFreq)),
            enable_baseline_correction=req.filterConfig.enableBaselineCorrection,
            enable_smoothing=req.filterConfig.enableSmoothing,
            smoothing_window=req.filterConfig.smoothingWindow,
        )
        filter_setup_time = time.time() - filter_setup_start
        logger.info(f"Filter setup completed in {filter_setup_time:.3f}s")

        # Process the ECG data using optimized methods
        processing_start = time.time()

        # Try vectorized processing first (fastest)
        try:
            logger.info("Attempting vectorized processing of ECG data")
            ecg_glove.decode_data(raw_bytes)
            logger.info("ECG data decoded successfully")

            # Check if vectorized processing is available
            if ecg_glove.is_vectorized_available():
                logger.info("Processing ECG data with vectorization")
                quality_results = ecg_glove.process_with_vectorization()
                logger.info("ECG data processed successfully with vectorization")
                processing_method = "vectorized"
            else:
                logger.warning("Vectorized processing not available, using optimized")
                quality_results = ecg_glove.process_optimized()
                logger.info("ECG data processed successfully with optimized method")
                processing_method = "optimized"

        except Exception as e:
            logger.warning(
                f"Primary processing failed: {e}, falling back to standard processing"
            )
            try:
                # Final fallback to standard processing
                logger.info("Attempting standard processing of ECG data")
                quality_results = ecg_glove.compute_quality()
                logger.info("ECG data processed successfully with standard method")
                processing_method = "standard"
            except Exception as e2:
                logger.error(f"All processing methods failed: {e2}")
                raise HTTPException(
                    status_code=500, detail=f"ECG processing failed: {str(e2)}"
                )

        processing_time = time.time() - processing_start
        logger.info(
            f"ECG data processing completed in {processing_time:.3f}s using {processing_method} method"
        )

        if not quality_results or not isinstance(quality_results, dict):
            logger.error(f"Invalid quality results: {quality_results}")
            raise HTTPException(status_code=500, detail="Failed to analyze ECG quality")

        # Extract data from the quality results - handle different formats from different processing methods
        if processing_method == "vectorized" or processing_method == "optimized":
            # For optimized methods, quality results are nested under 'quality' key
            lead_quality = quality_results.get("quality", {}).get("lead_quality", {})
            if not lead_quality:
                # Fallback: try direct access for backwards compatibility
                lead_quality = quality_results.get("lead_quality", {})
        else:
            # Standard processing returns quality directly
            lead_quality = quality_results.get("lead_quality", {})

        # Calculate overall quality from individual lead qualities
        overall_quality = 0.0
        if lead_quality:
            # Try to get overall quality from the results first
            overall_quality_sources = [
                quality_results.get("overall_quality"),
                quality_results.get("quality", {}).get("overall_quality"),
                quality_results.get("quality", {}).get("overall_quality_score"),
            ]

            for source in overall_quality_sources:
                if source is not None:
                    overall_quality = safe_float_conversion(source)
                    break
            else:
                # Fall back to calculating from individual lead qualities
                nk_qualities = []
                for lead_data in lead_quality.values():
                    if isinstance(lead_data, dict):
                        nk_quality = lead_data.get("nk_quality")
                        if nk_quality is not None:
                            nk_qualities.append(safe_float_conversion(nk_quality))

                overall_quality = (
                    sum(nk_qualities) / len(nk_qualities) if nk_qualities else 0.0
                )

        # Format the simplified response - return only grade and problems
        result = {
            "grade": get_quality_grade(overall_quality),
            "problems": extract_problems(lead_quality),
        }

        total_time = time.time() - start_time
        logger.info(
            f"ECG quality analysis completed successfully in {total_time:.3f}s using {processing_method} - Grade: {result['grade']}, Problems: {result['problems']}"
        )

        return result

    except ImportError as ie:
        error_msg = f"Missing Python dependencies: {str(ie)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error processing ECG quality analysis: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        # Return a fallback response instead of raising an exception
        result = {"grade": "bad", "problems": f"Analysis failed: {str(e)}"}
        return result


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": app.title, "version": app.version}


@app.post("/test_ecg_quality")
async def test_ecg_quality(req: EcgQualityRequest):
    """
    Test endpoint that returns mock ECG quality data for development/testing.
    """
    logger.info("Returning mock ECG quality data for testing")

    result = {"grade": "normal", "problems": "muscle movement interference"}

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
