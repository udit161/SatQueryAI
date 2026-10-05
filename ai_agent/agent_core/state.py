"""
SatQuery AI - Agent State Schema
Tracks the entire lifecycle of a satellite intelligence user request across the LangGraph state graph.
Includes rich schemas for:
- Inputs: Raw query, uploaded image paths (single, bi-temporal, optical-SAR), formats (GeoTIFF, COG, PNG).
- Image Modality Detection: Per-image TypedDict records capturing detected modality (optical, SAR, bi_temporal,
  multispectral), band count, sensor identity, acquisition time, and spatial role in the workflow.
- Thought Process: Classified task ("VQA", "change_detection", "grounding", "cross_modal_fusion"), selected
  specialist models, and reasoning traces.
- Multi-Step Execution: Chained compound query pipelines and execution queues.
- Validation Flags: Format support, modality compatibility, geospatial boundary checks, and error logs.
- Intermediate Tool Outputs: Named typed slots — vqa_answer, spatial_masks, bounding_boxes, change_mask,
  land_cover_labels, fusion_result — replacing opaque dict lookups in downstream nodes.
- Tool Confidence Scores: Per-specialist floating-point confidence registry.
- Auditable Execution Trace: Append-only list of ExecutionTraceEntry records logging tool name,
  parameters, status, confidence, latency, and error for every tool call in the pipeline.
- Outputs: Final textual answer, spatial outputs (bounding boxes, change masks, GeoJSON), and confidence scores.

Primary TypedDicts
------------------
- RSAgentState       : Comprehensive LangGraph state for the remote-sensing agent workflow.
- ImageModalityEntry : Per-image metadata record with detected modality and spatial role.
- ExecutionTraceEntry: Auditable per-tool-call log entry.
- IntermediateToolOutputs : Named intermediate result slots for all specialist tools.
- BoundingBoxEntry   : Lightweight TypedDict for a single grounding detection.
- SpatialMaskEntry   : Lightweight TypedDict for a change / segmentation mask reference.
"""

import sys
sys.setrecursionlimit(10000)

from enum import Enum
from typing import TypedDict, List, Dict, Optional, Any, Union, Annotated, Literal
import operator
import uuid
from datetime import datetime

# Optional Pydantic support with zero-dependency fallback
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    
    class _FieldInfo:
        def __init__(self, default=None, default_factory=None, description=""):
            self.default = default
            self.default_factory = default_factory
            self.description = description

        def get_value(self):
            if self.default_factory is not None:
                return self.default_factory()
            return self.default if self.default is not ... else None

    def Field(default=None, default_factory=None, description=""):
        return _FieldInfo(default=default, default_factory=default_factory, description=description)

    class BaseModel:
        def __init__(self, **kwargs):
            # 1. Initialize from class hierarchy defaults
            for cls in reversed(self.__class__.__mro__):
                for k, v in getattr(cls, "__dict__", {}).items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, _FieldInfo):
                        setattr(self, k, v.get_value())
                    elif not callable(v):
                        setattr(self, k, v)
            # 2. Apply kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            out = {}
            for k, v in self.__dict__.items():
                if k.startswith("_"):
                    continue
                if hasattr(v, "model_dump") and callable(v.model_dump):
                    out[k] = v.model_dump()
                elif isinstance(v, Enum):
                    out[k] = v.value
                elif isinstance(v, list):
                    out[k] = [
                        item.model_dump() if (hasattr(item, "model_dump") and callable(item.model_dump)) else (item.value if isinstance(item, Enum) else item)
                        for item in v
                    ]
                elif isinstance(v, dict):
                    out[k] = {
                        dk: (dv.model_dump() if (hasattr(dv, "model_dump") and callable(dv.model_dump)) else (dv.value if isinstance(dv, Enum) else dv))
                        for dk, dv in v.items()
                    }
                else:
                    out[k] = v
            return out


class ImageFormat(str, Enum):
    """Supported satellite and aerial imagery file formats."""
    GEOTIFF = "geotiff"           # .tif / .tiff (GeoTIFF)
    COG = "cog"                   # Cloud Optimized GeoTIFF
    PNG = "png"                   # Standard RGB / Mask
    JPEG = "jpeg"                 # Standard lossy image
    JP2 = "jp2"                   # JPEG 2000 (Sentinel-2 L1C/L2A)
    HDF5 = "hdf5"                 # Hierarchical Data Format (.h5 / .hdf5)
    NETCDF = "netcdf"             # Network Common Data Form (.nc)
    SAFE = "safe"                 # ESA Sentinel Standard Archive Format
    OTHER = "other"


class SensorModality(str, Enum):
    """Satellite sensor modality types."""
    OPTICAL = "optical"
    SAR = "sar"                   # Synthetic Aperture Radar
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    THERMAL = "thermal"
    DEM = "dem"                   # Digital Elevation Model
    UNKNOWN = "unknown"


class RequestStatus(str, Enum):
    """Lifecycle status of a user query in the agentic workflow."""
    PENDING = "pending"
    ROUTING = "routing"
    PLANNING = "planning"
    VALIDATING = "validating"
    FETCHING_DATA = "fetching_data"
    SPECIALIST_INFERENCE = "specialist_inference"
    FUSION = "fusion"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_USER_INPUT = "requires_user_input"


class TaskType(str, Enum):
    """Identified task category for satellite intelligence query."""
    VQA = "VQA"                                       # Visual Question Answering on Earth Observation data
    CHANGE_DETECTION = "change_detection"             # Bi-temporal environmental or structural changes
    GROUNDING = "grounding"                           # Spatial object localization and bounding box detection
    CROSS_MODAL_FUSION = "cross_modal_fusion"         # Optical + SAR all-weather multi-sensor fusion
    LAND_COVER_CLASSIFICATION = "land_cover"          # Pixel/segment land use classification
    DAMAGE_ASSESSMENT = "damage_assessment"           # Post-disaster structural/flood damage evaluation
    COMPOUND_PIPELINE = "compound_pipeline"           # Chained multi-specialist workflow (e.g. Fusion -> Change -> Grounding)
    GENERAL_EXPLORATION = "general_exploration"       # Geospatial search and general Q&A


class SpecialistModelType(str, Enum):
    """Catalog of available specialist models and backbones."""
    VISION_VQA_MODEL = "vision_vqa_model"             # BigEarthNet / GeoCLIP / ViT VQA Model
    CHANGE_DETECTOR_MODEL = "change_detector"         # Siamese Bi-temporal CNN/Transformer
    GROUNDING_RS_MODEL = "grounding_rs"               # Remote sensing open-vocabulary object detector
    CROSS_MODAL_FUSION_NET = "cross_modal_fusion"     # Optical-SAR cross-attention alignment module
    LAND_COVER_CLASSIFIER = "land_cover_classifier"   # Multi-spectral ResNet / ConvNeXt classifier
    SAR_FLOOD_MAPPER = "sar_flood_mapper"             # Sentinel-1 SAR water thresholding / UNet
    GEO_METADATA_EXTRACTOR = "geo_metadata_extractor" # GDAL / Rasterio metadata parser


# ---------------------------------------------------------------------------
# Validation & Thought Process Models
# ---------------------------------------------------------------------------

class ValidationFlags(BaseModel):
    """
    Granular gatekeeping flags verifying inputs, formats, coordinates, and task compatibility.
    """
    is_valid: bool = Field(True, description="Master validation status: False if any hard check fails")
    has_required_images: bool = Field(False, description="True if required images for the classified task are present")
    is_format_supported: bool = Field(True, description="True if image formats (GeoTIFF, COG, PNG, JP2) are supported")
    is_geospatial_valid: bool = Field(True, description="True if coordinates/BBox are within valid latitude/longitude bounds")
    is_modality_compatible: bool = Field(True, description="True if modalities match task requirements (e.g. T1+T2 for change detection)")
    is_temporal_ordered: bool = Field(True, description="True if T1 acquisition precedes or equals T2")
    is_resolution_sufficient: bool = Field(True, description="True if GSD is sufficient for target feature detection")
    requires_human_clarification: bool = Field(False, description="True if user prompt is ambiguous or parameters are missing")
    evidence_inconclusive: bool = Field(False, description="True if visual evidence for grounding or change detection was inconclusive after verification")
    validation_errors: List[str] = Field(default_factory=list, description="List of blocking validation error messages")
    validation_warnings: List[str] = Field(default_factory=list, description="Non-blocking warning messages (e.g. high cloud cover)")


class ReasoningStep(BaseModel):
    """
    Individual thought step captured during agent reasoning and multi-agent routing.
    """
    step_number: int = Field(1, description="Sequential thought step index")
    agent_name: str = Field("orchestrator", description="Agent or module producing the reasoning step")
    thought: str = Field(..., description="Internal chain-of-thought analysis")
    classified_task: Optional[str] = Field(None, description="Task intent deduced during this reasoning step")
    selected_specialist_models: List[str] = Field(default_factory=list, description="Specialist models selected in this step")
    action_taken: Optional[str] = Field(None, description="Action or tool called following this thought")
    critique_or_reflection: Optional[str] = Field(None, description="Self-reflection or validation check on previous output")
    confidence: Optional[float] = Field(None, description="Confidence score associated with this step (0.0 to 1.0)")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Image Input & Geospatial Models
# ---------------------------------------------------------------------------

class ImageInput(BaseModel):
    """
    Detailed metadata and path for an uploaded satellite image.
    Supports GeoTIFF, COG, JP2, and standard raster formats.
    """
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = Field(..., description="Local filesystem path or storage URI to the image file")
    file_format: ImageFormat = Field(ImageFormat.GEOTIFF, description="Image file format (e.g. GeoTIFF, PNG, COG)")
    modality: SensorModality = Field(SensorModality.OPTICAL, description="Sensor modality: optical, sar, multispectral")
    sensor_name: Optional[str] = Field(None, description="Satellite/Sensor source (e.g., Sentinel-2, Sentinel-1, Landsat-9, PlanetScope)")
    acquisition_timestamp: Optional[str] = Field(None, description="Image capture timestamp (ISO-8601)")
    bands: Optional[List[str]] = Field(default_factory=list, description="Band designations (e.g. ['B02', 'B03', 'B04', 'B08'])")
    resolution_meters: Optional[float] = Field(None, description="Spatial resolution / GSD in meters per pixel")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g. 'EPSG:4326', 'EPSG:32643')")
    bounds: Optional[List[float]] = Field(None, description="[min_lon, min_lat, max_lon, max_lat] spatial bounding box")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom header / driver metadata")


class BiTemporalImagePair(BaseModel):
    """
    Pre-event (T1) and post-event (T2) image pair for change detection and temporal analysis.
    """
    t1_image: ImageInput = Field(..., description="Time 1 (pre-event / baseline) satellite image")
    t2_image: ImageInput = Field(..., description="Time 2 (post-event / comparison) satellite image")
    temporal_gap_days: Optional[float] = Field(None, description="Calculated duration in days between T1 and T2 acquisitions")
    alignment_verified: bool = Field(False, description="Whether coregistration/orthorectification has been confirmed")


class OpticalSARImagePair(BaseModel):
    """
    Co-registered Optical and SAR imagery pair for cross-modal fusion and all-weather analysis.
    """
    optical_image: ImageInput = Field(..., description="Optical / multi-spectral satellite imagery")
    sar_image: ImageInput = Field(..., description="Synthetic Aperture Radar (SAR) imagery (e.g. Sentinel-1 GRD/SLC)")
    polarization: Optional[str] = Field(None, description="SAR polarization modes (e.g. 'VV+VH', 'HH+HV')")
    fusion_strategy: Optional[str] = Field("cross_attention", description="Fusion method: pixel_level, feature_fusion, cross_attention")


class GeoSpatialContext(BaseModel):
    """Geographic bounding box, coordinates, and temporal query criteria."""
    latitude: Optional[float] = Field(None, description="Center latitude coordinate")
    longitude: Optional[float] = Field(None, description="Center longitude coordinate")
    bbox: Optional[List[float]] = Field(None, description="[min_lon, min_lat, max_lon, max_lat] bounding box")
    zoom_level: Optional[int] = Field(None, description="Map zoom level (1-20)")
    crs: str = Field("EPSG:4326", description="Coordinate Reference System")
    date_start: Optional[str] = Field(None, description="Start date for temporal filtering (ISO-8601)")
    date_end: Optional[str] = Field(None, description="End date for temporal filtering (ISO-8601)")


class ModalityInputs(BaseModel):
    """
    Aggregated container for all uploaded image inputs.
    Accommodates single images, bi-temporal pairs, and optical-SAR pairs.
    """
    single_image: Optional[ImageInput] = Field(None, description="Primary single image for VQA or object grounding")
    bi_temporal_pair: Optional[BiTemporalImagePair] = Field(None, description="T1/T2 pair for bi-temporal change detection")
    optical_sar_pair: Optional[OpticalSARImagePair] = Field(None, description="Optical-SAR pair for cross-modal fusion")
    uploaded_images: List[ImageInput] = Field(default_factory=list, description="All uploaded image files associated with request")
    
    # Direct URI/path conveniences
    optical_image_url: Optional[str] = Field(None, description="Direct URL/path to optical image if available")
    sar_image_url: Optional[str] = Field(None, description="Direct URL/path to SAR image if available")
    t1_image_url: Optional[str] = Field(None, description="Direct URL/path to T1 image if available")
    t2_image_url: Optional[str] = Field(None, description="Direct URL/path to T2 image if available")
    multi_spectral_bands: Dict[str, str] = Field(default_factory=dict, description="Band file paths mapping")


# ---------------------------------------------------------------------------
# Structured Output Models (Spatial Outputs, Masks, Bounding Boxes, Confidence)
# ---------------------------------------------------------------------------

class BoundingBoxOutput(BaseModel):
    """
    Detected object / target grounding bounding box with spatial and geographic coordinates.
    """
    box_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str = Field(..., description="Target entity class (e.g. 'airplane', 'storage_tank', 'ship', 'building')")
    confidence: float = Field(..., description="Grounding confidence score (0.0 to 1.0)")
    bbox_normalized: List[float] = Field(..., description="[xmin, ymin, xmax, ymax] normalized coordinates (0.0 to 1.0)")
    bbox_pixels: Optional[List[int]] = Field(None, description="[xmin, ymin, xmax, ymax] pixel coordinates in source image")
    bbox_geo: Optional[List[float]] = Field(None, description="[min_lon, min_lat, max_lon, max_lat] WGS84 geographic coordinates")
    polygon_coordinates: Optional[List[List[float]]] = Field(None, description="Polygon boundary vertices if available")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Estimated dimensions, orientation, area in m²")


class ChangeMaskOutput(BaseModel):
    """
    Spatial mask for bi-temporal change detection, flood mapping, or deforestation analysis.
    """
    mask_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mask_type: str = Field("change_mask", description="Mask category: 'bi_temporal_change', 'deforestation', 'urban_expansion', 'flood'")
    mask_uri: str = Field(..., description="URI or local filesystem path to the generated mask GeoTIFF/PNG")
    changed_area_sq_km: Optional[float] = Field(None, description="Calculated total surface area of detected changes in km²")
    changed_area_pixels: Optional[int] = Field(None, description="Total count of positive change pixels")
    change_percentage: Optional[float] = Field(None, description="Percentage of analyzed AOI showing change")
    class_distribution: Dict[str, float] = Field(default_factory=dict, description="Breakdown of change categories and pixel ratios")
    color_map: Dict[str, str] = Field(default_factory=dict, description="Mapping of pixel values to hex colors for UI rendering")
    georeferencing: Dict[str, Any] = Field(default_factory=dict, description="CRS and affine transform parameters")


class SpatialOutputs(BaseModel):
    """
    Comprehensive container for all spatial analysis outputs.
    """
    bounding_boxes: List[BoundingBoxOutput] = Field(default_factory=list, description="Detected objects and bounding boxes")
    change_mask: Optional[ChangeMaskOutput] = Field(None, description="Bi-temporal change detection mask")
    segmentation_masks: List[ChangeMaskOutput] = Field(default_factory=list, description="Additional thematic segmentation masks")
    geojson_feature_collection: Optional[Dict[str, Any]] = Field(None, description="Ready-to-render GeoJSON for Mapbox/Leaflet UI")


class ConfidenceScores(BaseModel):
    """
    Detailed confidence and certainty breakdown across all executing models.
    """
    overall: float = Field(0.0, description="Master weighted confidence score (0.0 to 1.0)")
    vqa_confidence: Optional[float] = Field(None, description="Visual Question Answering confidence")
    grounding_confidence: Optional[float] = Field(None, description="Object detection / grounding confidence")
    change_confidence: Optional[float] = Field(None, description="Change detection classification confidence")
    fusion_confidence: Optional[float] = Field(None, description="Cross-modal alignment confidence")
    data_quality_score: Optional[float] = Field(None, description="Image quality, cloud clearance, and resolution adequacy score")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Model-specific confidence mappings")
    uncertainty_notes: Optional[str] = Field(None, description="Explanatory notes on sources of uncertainty")


class Artifact(BaseModel):
    """Generated visual or data artifacts resulting from agent execution."""
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: str = Field("visualization", description="Type of artifact: heatmap, mask, geojson, fused_image, chart, report")
    uri: str = Field("", description="URI or local path to the artifact file")
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionLog(BaseModel):
    """Log entry for an executed specialist tool or subagent."""
    tool_name: str = Field("", description="Name of the executed tool")
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    output_payload: Any = None
    execution_time_ms: Optional[float] = None
    status: str = "success"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Detected Modality Literal
# ---------------------------------------------------------------------------

# Exhaustive set of image modality roles recognised by the RS agent workflow.
# Used as the `detected_modality` field in ImageModalityEntry so graph nodes
# can pattern-match on a closed set without fragile string comparisons.
DetectedModality = Literal[
    "optical",          # Passive optical / multi-spectral (e.g. Sentinel-2, Landsat, PlanetScope)
    "sar",              # Synthetic Aperture Radar (e.g. Sentinel-1 GRD/SLC, ALOS-2)
    "bi_temporal",      # Single image participating in a before/after temporal pair
    "multispectral",    # Explicitly tagged multi-spectral raster (superset of optical)
    "hyperspectral",    # Hyperspectral cube (e.g. PRISMA, DESIS, HyMap)
    "thermal",          # Long-wave infrared / thermal band (e.g. TIRS, MODIS LST)
    "dem",              # Digital Elevation Model (e.g. SRTM, Copernicus DEM)
    "unknown",          # Could not be auto-detected from file metadata
]


# ---------------------------------------------------------------------------
# RS-Workflow TypedDicts
# ---------------------------------------------------------------------------

class ImageModalityEntry(TypedDict, total=False):
    """
    Per-image metadata record for a single uploaded satellite raster.

    Created by the validation / routing node after inspecting the uploaded file
    (via rasterio / GDAL metadata) and attached to RSAgentState.image_inputs.

    Fields
    ------
    image_id : str
        UUID assigned at upload time.
    image_path : str
        Absolute or relative filesystem path / S3 URI to the raster.
    detected_modality : DetectedModality
        Sensor modality auto-detected from band count, file name patterns, or
        explicit user tagging.  Drives task-routing decisions downstream.
    file_format : str
        File extension / driver name (e.g. 'geotiff', 'cog', 'jpeg2000', 'png').
    band_count : int
        Number of spectral bands in the raster (e.g. 1 = panchromatic, 12 = S2-L2A).
    band_names : List[str]
        Ordered list of band designations (e.g. ['B02', 'B03', 'B04', 'B08']).
        Empty list if unknown.
    spatial_role : str
        Logical role within the current task:
        'primary'  – single image for VQA / grounding,
        'optical'  – optical half of an Optical-SAR pair,
        'sar'      – SAR half of an Optical-SAR pair,
        't1'       – pre-event image in a bi-temporal pair,
        't2'       – post-event image in a bi-temporal pair,
        'auxiliary'– supplementary reference image.
    sensor_name : str
        Satellite / instrument identifier (e.g. 'Sentinel-2', 'Sentinel-1', 'Landsat-9').
    acquisition_timestamp : str
        ISO-8601 acquisition datetime (e.g. '2024-06-15T10:32:00Z').  Empty string if unknown.
    resolution_meters : float
        Native ground-sampling distance in metres per pixel (0.0 if unknown).
    crs : str
        EPSG string of the native coordinate reference system (e.g. 'EPSG:32643').
        Empty string if the file carries no CRS (e.g. plain PNG).
    bounds_wgs84 : List[float]
        [min_lon, min_lat, max_lon, max_lat] in WGS-84.  Empty list if unknown.
    cloud_cover_pct : float
        Estimated scene cloud-cover percentage [0.0 – 100.0].  -1.0 if not applicable / unknown.
    is_georeferenced : bool
        True if the file carries a valid CRS and affine transform.
    metadata : Dict[str, Any]
        Freeform driver / EXIF / STAC metadata parsed from the file header.
    """
    image_id: str
    image_path: str
    detected_modality: DetectedModality
    file_format: str
    band_count: int
    band_names: List[str]
    spatial_role: str
    sensor_name: str
    acquisition_timestamp: str
    resolution_meters: float
    crs: str
    bounds_wgs84: List[float]
    cloud_cover_pct: float
    is_georeferenced: bool
    metadata: Dict[str, Any]


class ExecutionTraceEntry(TypedDict, total=False):
    """
    Auditable record of a single tool invocation in the agent pipeline.

    One entry is appended to RSAgentState.execution_trace every time a
    specialist tool or subagent is called, regardless of success or failure.
    The trace is append-only (LangGraph operator.add reducer) so the full
    call history is preserved across parallel / branching graph nodes.

    Fields
    ------
    trace_id : str
        UUID for this trace entry (for cross-referencing with logs / DB).
    tool_name : str
        Canonical identifier of the called tool
        (e.g. 'vqa_tool', 'grounding_tool', 'change_detection_tool',
        'fusion_routing_tool', 'land_cover_tool').
    node_name : str
        LangGraph node that triggered the call (e.g. 'vqa_specialist_node').
    parameters : Dict[str, Any]
        Exact input parameters passed to the tool at call time.  Serialisable
        to JSON; large tensors should be replaced with a shape/dtype summary.
    status : str
        Execution outcome: 'success' | 'error' | 'skipped' | 'timeout'.
    result_summary : str
        One-sentence human-readable summary of the tool output or error reason.
    confidence : float
        Tool-reported confidence score [0.0 – 1.0].  0.0 on error or skipped.
    duration_ms : float
        Wall-clock execution time in milliseconds.
    timestamp_start : str
        ISO-8601 UTC timestamp when the tool was invoked.
    timestamp_end : str
        ISO-8601 UTC timestamp when the tool returned.
    error : str
        Error message or exception traceback if status == 'error'.  Empty string otherwise.
    output_keys : List[str]
        List of top-level keys present in the tool's output dictionary.  Useful
        for quick schema inspection without loading the full output.
    """
    trace_id: str
    tool_name: str
    node_name: str
    parameters: Dict[str, Any]
    status: str
    result_summary: str
    confidence: float
    duration_ms: float
    timestamp_start: str
    timestamp_end: str
    error: str
    output_keys: List[str]


class BoundingBoxEntry(TypedDict, total=False):
    """
    Lightweight TypedDict representation of a single grounding detection.

    Mirrors the dict schema returned by VisionVQAModel._generate_bounding_boxes()
    and stored in IntermediateToolOutputs.bounding_boxes.

    Fields
    ------
    box_id : str           UUID for this detection.
    label : str            Target entity class label.
    confidence : float     Localization confidence [0.0 – 1.0].
    bbox_normalized : List[float]
        [xmin, ymin, xmax, ymax] in [0, 1] relative to image W/H.
    bbox_pixels : List[int]
        [xmin, ymin, xmax, ymax] in source-image pixel coordinates.
    bbox_geo : List[float]
        [min_lon, min_lat, max_lon, max_lat] in WGS-84.  Empty list if CRS unavailable.
    polygon_coordinates : List[List[float]]
        Polygon ring vertices [[lon, lat], ...].  Empty list if not computed.
    attributes : Dict[str, Any]
        Per-box metrics: area_normalized, area_sq_m, orientation_deg, etc.
    source_tool : str
        Name of the tool that produced this box (e.g. 'grounding_tool').
    image_id : str
        ID of the source image this box was detected in.
    """
    box_id: str
    label: str
    confidence: float
    bbox_normalized: List[float]
    bbox_pixels: List[int]
    bbox_geo: List[float]
    polygon_coordinates: List[List[float]]
    attributes: Dict[str, Any]
    source_tool: str
    image_id: str


class SpatialMaskEntry(TypedDict, total=False):
    """
    Lightweight TypedDict reference to a spatial mask output (change mask,
    segmentation mask, flood extent, etc.).

    Fields
    ------
    mask_id : str           UUID for this mask artifact.
    mask_type : str
        Category: 'bi_temporal_change' | 'deforestation' | 'urban_expansion' |
        'flood' | 'land_cover_segmentation' | 'damage_extent' | 'custom'.
    mask_uri : str          Filesystem path or S3 URI to the mask GeoTIFF/PNG.
    changed_area_sq_km : float   Surface area of positive-class pixels in km².
    changed_area_pixels : int    Count of positive-class pixels.
    change_percentage : float    Positive-class fraction of the analysed AOI [0 – 100].
    class_distribution : Dict[str, float]
        Breakdown of change sub-classes and their pixel-area fractions.
    color_map : Dict[str, str]
        Pixel-value → HEX-colour mapping for UI rendering (e.g. {'1': '#FF3333'}).
    georeferencing : Dict[str, Any]
        CRS string, affine transform coefficients, and source image IDs.
    confidence : float      Mask-level confidence score [0.0 – 1.0].
    source_tool : str       Tool that generated this mask (e.g. 'change_detection_tool').
    """
    mask_id: str
    mask_type: str
    mask_uri: str
    changed_area_sq_km: float
    changed_area_pixels: int
    change_percentage: float
    class_distribution: Dict[str, float]
    color_map: Dict[str, str]
    georeferencing: Dict[str, Any]
    confidence: float
    source_tool: str


class IntermediateToolOutputs(TypedDict, total=False):
    """
    Named intermediate result slots for all specialist tools in the RS pipeline.

    Replaces the opaque ``intermediate_outputs: Dict[str, Any]`` field in
    ``AgentState`` with explicit, typed keys so that downstream nodes can
    access results without string-key guessing or casting.

    All fields are optional (total=False) because a given workflow run will
    only populate the slots relevant to its task type.

    Fields
    ------
    vqa_answer : str
        Textual answer produced by the VQA specialist (vqa_tool).
    vqa_confidence : float
        Confidence score for the vqa_answer [0.0 – 1.0].
    vqa_embedding_shape : List[int]
        Shape of the visual feature embedding used for the answer (diagnostic).
    bounding_boxes : List[BoundingBoxEntry]
        All grounding detections across every image in the current task,
        aggregated from grounding_tool invocations.
    spatial_masks : List[SpatialMaskEntry]
        All spatial masks produced during the run (change masks, flood masks,
        segmentation masks).  Each entry links to its artifact file URI.
    change_mask : SpatialMaskEntry
        Primary bi-temporal change mask (convenience alias; also present in
        spatial_masks).  None if no change-detection step was run.
    land_cover_labels : Dict[str, float]
        Land-cover class → percentage-of-AOI mapping from land_cover_tool.
        e.g. {'Mixed Forest': 34.1, 'Cropland': 28.7, 'Urban': 19.3, ...}
    land_cover_confidence : float
        Overall land-cover classification confidence [0.0 – 1.0].
    fusion_result : Dict[str, Any]
        Full output dictionary from fusion_routing_tool (Optical-SAR fusion).
        Includes aligned_multimodal_shape, alignment_score, cloud_penetration_index.
    fusion_confidence : float
        Cross-modal alignment confidence score [0.0 – 1.0].
    damage_assessment : Dict[str, Any]
        Structured output from damage_assessment_tool (post-disaster analysis).
    general_answer : str
        Fallback free-text answer for GENERAL_EXPLORATION tasks.
    raw_tool_outputs : Dict[str, Any]
        Verbatim output dicts keyed by tool_name, preserved for debugging.
        e.g. {'vqa_tool': {...}, 'grounding_tool': {...}}
    """
    vqa_answer: str
    vqa_confidence: float
    vqa_embedding_shape: List[int]
    bounding_boxes: List[BoundingBoxEntry]
    spatial_masks: List[SpatialMaskEntry]
    change_mask: SpatialMaskEntry
    land_cover_labels: Dict[str, float]
    land_cover_confidence: float
    fusion_result: Dict[str, Any]
    fusion_confidence: float
    damage_assessment: Dict[str, Any]
    general_answer: str
    raw_tool_outputs: Dict[str, Any]


# ---------------------------------------------------------------------------
# Conversational Memory Schemas
# ---------------------------------------------------------------------------

class ConversationTurn(TypedDict, total=False):
    """
    Immutable, append-only record of one completed agent interaction turn.

    Written by ``aggregation_node`` at the end of every graph run and pushed
    onto ``AgentState.conversation_history`` (operator.add reducer).  Future
    turns read this list to resolve relative back-references such as
    "that bounding box", "the same spot", or "the SAR image".

    Fields
    ------
    turn_id : str
        UUID assigned when the turn record is created.
    turn_index : int
        Zero-based sequential index across the session lifetime.
    raw_query : str
        Original user query text for this turn.
    classified_task : str
        Task routed to (e.g. 'grounding', 'change_detection', 'vqa').
    final_response : str
        Synthesized natural-language answer returned to the user.
    bounding_boxes : List[Dict]
        All bounding boxes produced during this turn (label, confidence,
        bbox_normalized, bbox_geo fields mirroring BoundingBoxEntry).
    spatial_masks : List[Dict]
        All change/segmentation masks produced (mirroring SpatialMaskEntry).
    image_ids : List[str]
        image_id values of all images used as input in this turn.
    image_paths : List[str]
        Filesystem / URI paths to all input images used in this turn.
    active_roi : Optional[Dict]
        Convenience alias for the single highest-confidence bounding box,
        pre-computed for fast reference resolution in subsequent turns.
    tool_names_used : List[str]
        Canonical names of every tool called during this turn, in order.
    confidence : float
        Overall confidence score for this turn [0.0 – 1.0].
    timestamp : str
        ISO-8601 UTC timestamp when the turn's aggregation_node completed.
    """
    turn_id: str
    turn_index: int
    raw_query: str
    classified_task: str
    final_response: str
    bounding_boxes: List[Dict[str, Any]]
    spatial_masks: List[Dict[str, Any]]
    image_ids: List[str]
    image_paths: List[str]
    active_roi: Optional[Dict[str, Any]]
    tool_names_used: List[str]
    confidence: float
    timestamp: str


class SpatialContextCache(TypedDict, total=False):
    """
    Session-level rolling index of all spatial features seen across turns.

    Updated in-place (merge_dicts reducer) by ``aggregation_node`` at the end
    of each turn, so the *latest_* slots always reflect the most recent turn's
    outputs.  The reference resolver in ``interpret_and_validate_node`` reads
    this cache for O(1) lookups without scanning the full history list.

    Fields
    ------
    latest_bounding_boxes : List[Dict]
        Bounding boxes produced in the most recent grounding or VQA turn.
    latest_masks : List[Dict]
        Change / segmentation masks from the most recent change-detection turn.
    latest_image_paths : Dict[str, str]
        Image paths keyed by spatial role:
        {'primary': path, 'optical': path, 'sar': path, 't1': path, 't2': path}.
    latest_image_ids : List[str]
        Image IDs used in the most recent turn.
    active_roi : Optional[Dict]
        Single "hot" bounding box the user is most likely referring to in a
        follow-up.  Set to the highest-confidence box from the latest grounding
        or VQA turn.  None if no boxes have been produced yet.
    last_task : str
        Task type of the most recently completed turn (e.g. 'grounding').
    turn_count : int
        Total number of successfully completed turns in this session.
    """
    latest_bounding_boxes: List[Dict[str, Any]]
    latest_masks: List[Dict[str, Any]]
    latest_image_paths: Dict[str, str]
    latest_image_ids: List[str]
    active_roi: Optional[Dict[str, Any]]
    last_task: str
    turn_count: int


# ---------------------------------------------------------------------------
# Reducer Helper Functions for LangGraph
# ---------------------------------------------------------------------------

def merge_dicts(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer function to shallow-merge dictionary updates across multiple specialist nodes."""
    merged = dict(a or {})
    if b and isinstance(b, dict):
        merged.update(b)
    return merged


def merge_intermediate_outputs(
    a: Optional[Dict[str, Any]],
    b: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Deep-merge reducer for IntermediateToolOutputs dicts.

    Rules:
    - List fields (bounding_boxes, spatial_masks) are *concatenated*, not replaced.
    - Dict fields (land_cover_labels, fusion_result, raw_tool_outputs) are shallow-merged.
    - Scalar fields (vqa_answer, vqa_confidence, ...) are replaced by the incoming value
      if it is non-None / non-empty.
    """
    merged: Dict[str, Any] = dict(a or {})
    if not b or not isinstance(b, dict):
        return merged

    LIST_FIELDS = {"bounding_boxes", "spatial_masks", "vqa_embedding_shape"}
    DICT_FIELDS = {"land_cover_labels", "fusion_result", "damage_assessment", "raw_tool_outputs"}

    for key, val in b.items():
        if key in LIST_FIELDS:
            existing = merged.get(key) or []
            merged[key] = list(existing) + (list(val) if val else [])
        elif key in DICT_FIELDS:
            existing = merged.get(key) or {}
            if isinstance(val, dict):
                merged[key] = {**existing, **val}
            else:
                merged[key] = val
        else:
            # Scalar: replace only if the incoming value is not None / empty string
            if val is not None and val != "":
                merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# LangGraph TypedDict State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Main state schema for LangGraph agent workflow.
    Tracks state transitions, multi-agent reasoning trajectory, thought process,
    selected specialist models, validation flags, intermediate model outputs, spatial outputs,
    confidence scores, multi-step pipeline execution, and final textual answers.

    Note: For new RS-workflow node implementations, prefer RSAgentState which provides
    explicit ImageModalityEntry, ExecutionTraceEntry, and IntermediateToolOutputs slots.
    """
    # 1. Identity & Session
    request_id: str
    user_id: Optional[str]
    session_id: Optional[str]

    # 2. Input Data & Raw User Prompts
    raw_query: str                                         # Exact raw prompt entered by the user
    query: str                                             # Processed / normalized query
    geo_context: Optional[Dict[str, Any]]

    # Image Inputs Tracking (Paths, Formats, and Pairs)
    single_image: Optional[Dict[str, Any]]                 # Single image path & format (e.g. GeoTIFF)
    bi_temporal_pair: Optional[Dict[str, Any]]             # T1 & T2 image paths, timestamps & formats
    optical_sar_pair: Optional[Dict[str, Any]]             # Optical & SAR paths, polarizations & formats
    uploaded_images: List[Dict[str, Any]]                  # Complete list of uploaded image inputs
    modalities: Optional[Dict[str, Any]]                   # Aggregated modality map

    # 3. Thought Process & Task Classification
    classified_task: Optional[str]                         # "VQA", "change_detection", "grounding", "cross_modal_fusion", "compound_pipeline"
    task_classification_confidence: Optional[float]        # Confidence score of task router (0.0 to 1.0)
    task_reasoning: Optional[str]                          # Rationale for selecting this task
    thought_trace: Annotated[List[Dict[str, Any]], operator.add]  # Cumulative reasoning/thought steps
    plan_steps: Optional[List[str]]
    current_step: int

    # 4. Multi-Step Execution Queue & Specialist Selection
    selected_specialist_models: List[str]                  # Sequence of models chosen: ["cross_modal_fusion", "change_detector", "grounding_rs"]
    execution_queue: List[str]                             # Pending specialists in compound pipeline
    completed_specialists: Annotated[List[str], operator.add] # Completed specialists
    is_compound_task: bool                                 # True if multiple specialists execute in sequence
    specialist_model_configs: Annotated[Dict[str, Any], merge_dicts] # Hyperparameters, thresholds, and runtime flags
    # Guardrail outputs: sanitized params written by validate_tool_params node
    sanitized_tool_params: Annotated[Dict[str, Any], merge_dicts]    # Final cleaned param map per tool, keyed by tool config key
    param_guardrail_log: Annotated[List[Dict[str, Any]], operator.add] # Append-only log of every param mutation event

    # 5. Validation Flags & Gatekeeping
    validation_flags: Annotated[Dict[str, Any], merge_dicts] # Granular checks (has_images, is_geospatial_valid, etc.)
    is_valid: bool                                         # Master gate: False stops pipeline and alerts user
    validation_errors: List[str]                           # List of blocking validation errors
    validation_warnings: List[str]                         # Non-blocking warnings
    requires_clarification: bool                           # True if human-in-the-loop clarification is needed
    evidence_inconclusive: bool                            # True if visual evidence was inconclusive after retry
    evidence_retry_count: int                              # Retry count for visual evidence verification

    # 6. Trajectory & LangGraph Message Reducers
    messages: Annotated[List[Dict[str, Any]], operator.add]
    active_agent: str
    routing_history: Annotated[List[str], operator.add]

    # 7. Tool & Model Intermediate Outputs
    tool_logs: Annotated[List[Dict[str, Any]], operator.add]
    intermediate_outputs: Annotated[Dict[str, Any], merge_dicts] # Merged dictionary of all specialist inferences

    # 8. Output Tracking: Textual Answers, Spatial Outputs & Confidence Scores
    final_response: Optional[str]                          # Final synthesized natural language answer
    executive_summary: Optional[str]                       # Brief summary for dashboards & notifications
    detailed_analysis: Optional[str]                       # Deep breakdown citing sensor evidence

    # Spatial Outputs
    spatial_outputs: Annotated[Dict[str, Any], merge_dicts] # Unified spatial results container
    bounding_boxes: Annotated[List[Dict[str, Any]], operator.add] # Bounding box coordinates & labels
    change_mask: Optional[Dict[str, Any]]                  # Change detection mask & area metrics

    # Confidence Metrics
    confidence_score: Optional[float]                      # Overall confidence score (0.0 to 1.0)
    confidence_breakdown: Optional[Dict[str, Any]]         # Per-specialist confidence scores & uncertainty

    # Artifacts (Downloadables, Overlays, Reports)
    artifacts: Annotated[List[Dict[str, Any]], operator.add]

    # 9. Lifecycle Status & Errors
    status: str
    error: Optional[str]

    # 10. Multi-Turn Conversational Memory
    # conversation_history: append-only list of ConversationTurn records (one per completed turn).
    # spatial_context_cache: rolling dict updated each turn with latest spatial outputs.
    # resolved_roi/resolved_image_path: injected by interpret_and_validate when resolving back-refs.
    # is_followup_query: True when relative references were detected and successfully resolved.
    # reference_resolution_log: append-only audit of every reference resolution event.
    conversation_history: Annotated[List[Dict[str, Any]], operator.add]  # ConversationTurn records
    spatial_context_cache: Annotated[Dict[str, Any], merge_dicts]        # SpatialContextCache (rolling)
    resolved_roi: Optional[Dict[str, Any]]                               # Resolved bounding box / ROI from prior turn
    resolved_image_path: Optional[str]                                    # Resolved image path from prior turn  
    is_followup_query: bool                                               # True when relative refs were resolved
    reference_resolution_log: Annotated[List[Dict[str, Any]], operator.add]  # Audit of reference resolutions


# ---------------------------------------------------------------------------
# RSAgentState — Comprehensive TypedDict for the RS Agent Workflow
# ---------------------------------------------------------------------------

class RSAgentState(TypedDict, total=False):
    """
    Comprehensive LangGraph TypedDict state for the SatQuery AI remote-sensing
    agent workflow.

    Designed to be the *single source of truth* passed between all graph nodes.
    It extends the capabilities of AgentState with:

    1.  **Explicit image modality tracking** via image_inputs (List[ImageModalityEntry]).
        Each element records the image path, auto-detected sensor modality, band count,
        spatial role (primary / optical / sar / t1 / t2), and raster metadata so that
        routing nodes can make task decisions without re-opening files.

    2.  **Named intermediate tool output slots** via tool_outputs (IntermediateToolOutputs).
        Replaces the opaque Dict[str, Any] with explicit typed fields:
        vqa_answer, bounding_boxes, spatial_masks, change_mask, land_cover_labels,
        fusion_result, etc.  Downstream nodes read these by name, not by string key.

    3.  **Per-tool confidence registry** via tool_confidence_scores Dict[str, float].
        Maps tool canonical name to its last-reported confidence score.  Used by
        the synthesis node to compute weighted final confidence.

    4.  **Auditable execution trace** via execution_trace (List[ExecutionTraceEntry]).
        Append-only list (LangGraph operator.add reducer) capturing every tool call:
        tool name, exact input parameters, status, result summary, confidence,
        wall-clock latency, and error details.

    LangGraph Reducer Annotations
    ------------------------------
    - execution_trace   : operator.add  — appended by every node, never overwritten.
    - thought_trace     : operator.add  — appended reasoning steps.
    - messages          : operator.add  — LangChain message history.
    - routing_history   : operator.add  — appended route names.
    - tool_logs         : operator.add  — raw ToolExecutionLog dicts.
    - completed_specialists : operator.add — accumulates finished model names.
    - tool_outputs      : merge_intermediate_outputs — deep-merges list concatenation
                          and dict shallow-merge per field type.
    - validation_flags  : merge_dicts   — last-write-wins shallow merge.
    - specialist_model_configs : merge_dicts — last-write-wins shallow merge.

    Usage Example
    -------------
    >>> state: RSAgentState = make_empty_rs_state(
    ...     raw_query="Detect deforestation in this Sentinel-2 image.",
    ...     request_id=str(uuid.uuid4())
    ... )
    >>> img_entry = make_image_entry(
    ...     image_path="/data/S2_20240615_B02B03B04.tif",
    ...     detected_modality="optical",
    ...     spatial_role="primary",
    ...     band_count=12,
    ...     sensor_name="Sentinel-2"
    ... )
    >>> state["image_inputs"].append(img_entry)
    """

    # ── Identity & Session ────────────────────────────────────────────────────
    request_id: str                   # UUID for the current request
    user_id: Optional[str]            # Authenticated user identifier
    session_id: Optional[str]         # Browser / API session identifier

    # ── Raw Query & Processed Query ───────────────────────────────────────────
    raw_query: str
    # Exact natural-language prompt submitted by the user without any preprocessing.
    # Preserved verbatim throughout the entire pipeline for auditability.

    query: str
    # Normalised / expanded version of raw_query produced by the routing node
    # (e.g. typo correction, entity disambiguation, query decomposition).

    geo_context: Optional[Dict[str, Any]]
    # GeoSpatialContext dict: center lat/lon, bbox [min_lon, min_lat, max_lon, max_lat],
    # CRS, and optional temporal filters (date_start / date_end).

    # ── Image Inputs with Detected Modalities ─────────────────────────────────
    image_inputs: List[ImageModalityEntry]
    # Ordered list of all uploaded image records, one per file.  Populated by
    # the validation / routing node after inspecting raster metadata.
    # Each entry carries: image_path, detected_modality, band_count, spatial_role,
    # sensor_name, acquisition_timestamp, resolution_meters, crs, bounds_wgs84,
    # cloud_cover_pct, is_georeferenced, and free-form metadata.
    #
    # Routing heuristics:
    #   len == 1 & modality in {optical, multispectral}  -> VQA or grounding
    #   len == 2 & roles == {t1, t2}                     -> change_detection
    #   len == 2 & roles == {optical, sar}               -> cross_modal_fusion
    #   any(modality == bi_temporal)                     -> change_detection

    # ── Task Classification & Intent ──────────────────────────────────────────
    classified_task: Optional[str]
    # Resolved task category string from the TaskType enum:
    # 'VQA' | 'change_detection' | 'grounding' | 'cross_modal_fusion' |
    # 'land_cover' | 'damage_assessment' | 'compound_pipeline' | 'general_exploration'

    task_classification_confidence: Optional[float]
    # Router confidence in the above classification [0.0 – 1.0].
    # Values below 0.60 trigger a clarification request to the user.

    task_reasoning: Optional[str]
    # Chain-of-thought rationale produced by the task-routing LLM or heuristic,
    # explaining why this task type was selected.

    thought_trace: Annotated[List[Dict[str, Any]], operator.add]
    # Cumulative log of ReasoningStep dicts appended by every graph node.
    # Each step records: step_number, agent_name, thought, action_taken,
    # classified_task, selected_specialist_models, confidence, timestamp.

    plan_steps: Optional[List[str]]
    # Ordered list of planned execution steps for compound pipelines.
    # e.g. ['fusion_routing_tool', 'change_detection_tool', 'grounding_tool']

    current_step: int
    # Zero-indexed pointer into plan_steps indicating the currently executing step.

    # ── Specialist Model Selection & Execution Queue ───────────────────────────
    selected_specialist_models: List[str]
    # Names of specialist models chosen for this request, in execution order.
    # e.g. ['vision_vqa_model', 'grounding_rs']

    execution_queue: List[str]
    # Remaining specialist model names yet to be executed in the current pipeline.

    completed_specialists: Annotated[List[str], operator.add]
    # Accumulates the names of successfully completed specialist models / tools.

    is_compound_task: bool
    # True when more than one specialist executes sequentially (compound pipeline).

    specialist_model_configs: Annotated[Dict[str, Any], merge_dicts]
    # Per-specialist hyperparameters and runtime overrides keyed by model name.
    # e.g. {'grounding_tool': {'box_threshold': 0.35, 'text_threshold': 0.25}}

    # ── Validation & Gatekeeping ───────────────────────────────────────────────
    validation_flags: Annotated[Dict[str, Any], merge_dicts]
    # ValidationFlags dict: is_valid, has_required_images, is_format_supported,
    # is_geospatial_valid, is_modality_compatible, is_temporal_ordered,
    # is_resolution_sufficient, requires_human_clarification.

    is_valid: bool
    # Master gate flag. If False, the pipeline halts and returns an error response.

    validation_errors: List[str]
    # Blocking error messages (e.g. 'GeoTIFF CRS missing', 'T2 precedes T1').

    validation_warnings: List[str]
    # Non-blocking warnings (e.g. 'Cloud cover > 30%', 'Resolution coarser than 10m').

    requires_clarification: bool
    # True if the orchestrator needs additional user input before proceeding.

    evidence_inconclusive: bool
    # True if visual evidence for grounding or change detection was inconclusive after verification retry.

    evidence_retry_count: int
    # Counter tracking visual evidence verification retries.

    # ── LangGraph Message Bus ─────────────────────────────────────────────────
    messages: Annotated[List[Dict[str, Any]], operator.add]
    # LangChain / LangGraph message history (HumanMessage, AIMessage, ToolMessage).

    active_agent: str
    # Name of the currently-executing agent node (e.g. 'vqa_specialist_node').

    routing_history: Annotated[List[str], operator.add]
    # Append-only list of graph-edge routing decisions in chronological order.

    # ── Auditable Execution Trace ─────────────────────────────────────────────
    execution_trace: Annotated[List[ExecutionTraceEntry], operator.add]
    # Append-only audit log of every tool invocation in the pipeline.
    # Each ExecutionTraceEntry records:
    #   trace_id         – UUID
    #   tool_name        – canonical tool identifier
    #   node_name        – LangGraph node that called the tool
    #   parameters       – exact input parameters dict (JSON-serialisable)
    #   status           – 'success' | 'error' | 'skipped' | 'timeout'
    #   result_summary   – one-sentence human-readable outcome
    #   confidence       – tool-reported confidence [0.0 – 1.0]
    #   duration_ms      – wall-clock latency in milliseconds
    #   timestamp_start  – ISO-8601 UTC invocation time
    #   timestamp_end    – ISO-8601 UTC return time
    #   error            – exception message / traceback on failure
    #   output_keys      – top-level keys present in the tool's output dict

    tool_logs: Annotated[List[Dict[str, Any]], operator.add]
    # Raw ToolExecutionLog dicts (legacy; execution_trace is the preferred audit trail).

    # ── Intermediate Tool Outputs (named slots) ────────────────────────────────
    tool_outputs: Annotated[IntermediateToolOutputs, merge_intermediate_outputs]
    # Named typed slots for results from each specialist tool.  Updated by each
    # specialist node via merge_intermediate_outputs reducer (list concatenation
    # for bounding_boxes / spatial_masks; shallow dict merge for label maps).
    #
    # Key slots:
    #   vqa_answer         – textual answer from vqa_tool
    #   vqa_confidence     – VQA confidence score
    #   bounding_boxes     – list of BoundingBoxEntry dicts from grounding_tool
    #   spatial_masks      – list of SpatialMaskEntry dicts from change_detection_tool
    #   change_mask        – primary bi-temporal change mask (SpatialMaskEntry)
    #   land_cover_labels  – {class: pct} map from land_cover_tool
    #   fusion_result      – full output dict from fusion_routing_tool
    #   raw_tool_outputs   – verbatim tool dicts keyed by tool_name

    intermediate_outputs: Annotated[Dict[str, Any], merge_dicts]
    # Legacy opaque dict retained for backward compat with tools.py callers.
    # New code should use tool_outputs instead.

    # ── Per-Tool Confidence Scores ────────────────────────────────────────────
    tool_confidence_scores: Dict[str, float]
    # Maps canonical tool name to its most recently reported confidence score.
    # Updated by each specialist node after every tool call.
    # e.g. {'vqa_tool': 0.94, 'grounding_tool': 0.91, 'change_detection_tool': 0.88}
    # The synthesis node computes a weighted mean across all present scores.

    # ── Output: Textual Answers ───────────────────────────────────────────────
    final_response: Optional[str]
    # Final synthesized natural-language answer returned to the user.

    executive_summary: Optional[str]
    # Brief one-paragraph summary for dashboard cards and push notifications.

    detailed_analysis: Optional[str]
    # Extended analytical breakdown citing band-specific evidence, metrics,
    # and confidence intervals.

    # ── Output: Spatial Results ───────────────────────────────────────────────
    spatial_outputs: Annotated[Dict[str, Any], merge_dicts]
    # Unified spatial results container (bounding_boxes list + change_mask dict
    # + segmentation_masks list + geojson_feature_collection).

    bounding_boxes: Annotated[List[Dict[str, Any]], operator.add]
    # All detected bounding boxes aggregated across the pipeline.
    # Each dict follows the BoundingBoxEntry schema.

    change_mask: Optional[Dict[str, Any]]
    # Primary bi-temporal change mask.  Follows the SpatialMaskEntry schema.

    # ── Output: Confidence Metrics ────────────────────────────────────────────
    confidence_score: Optional[float]
    # Master weighted confidence score [0.0 – 1.0] computed by the synthesis node
    # as a weighted mean of tool_confidence_scores.

    confidence_breakdown: Optional[Dict[str, Any]]
    # Per-specialist and per-data-quality confidence breakdown (ConfidenceScores dict).

    # ── Output: Artifacts ────────────────────────────────────────────────────
    artifacts: Annotated[List[Dict[str, Any]], operator.add]
    # Downloadable output files (mask GeoTIFFs, GeoJSON overlays, PDF reports).
    # Each entry follows the Artifact schema: artifact_id, artifact_type, uri, title.

    # ── Lifecycle ────────────────────────────────────────────────────────────
    status: str
    # RequestStatus value: 'pending' | 'routing' | 'validating' | 'specialist_inference'
    # | 'synthesizing' | 'completed' | 'failed' | 'requires_user_input'

    error: Optional[str]
    # Human-readable error message if status == 'failed'.

    created_at: str
    # ISO-8601 UTC timestamp when this request was first created.

    updated_at: str
    # ISO-8601 UTC timestamp of the most recent state mutation.


# ---------------------------------------------------------------------------
# RSAgentState Factory Helpers
# ---------------------------------------------------------------------------

def make_image_entry(
    image_path: str,
    detected_modality: DetectedModality = "unknown",
    spatial_role: str = "primary",
    band_count: int = 0,
    band_names: Optional[List[str]] = None,
    sensor_name: str = "",
    acquisition_timestamp: str = "",
    resolution_meters: float = 0.0,
    crs: str = "",
    bounds_wgs84: Optional[List[float]] = None,
    cloud_cover_pct: float = -1.0,
    is_georeferenced: bool = False,
    file_format: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> ImageModalityEntry:
    """
    Construct a fully-populated ImageModalityEntry dict.

    All fields have sensible defaults so callers only need to supply the
    mandatory image_path and the fields they know at construction time.

    Args:
        image_path:            Absolute filesystem path or S3 URI.
        detected_modality:     Auto-detected sensor modality (DetectedModality literal).
        spatial_role:          Logical role: 'primary' | 'optical' | 'sar' | 't1' | 't2'.
        band_count:            Number of spectral bands (0 = unknown).
        band_names:            Ordered band labels (e.g. ['B02', 'B03', 'B04', 'B08']).
        sensor_name:           Satellite / instrument name (e.g. 'Sentinel-2').
        acquisition_timestamp: ISO-8601 capture time (empty string = unknown).
        resolution_meters:     Native GSD in metres/pixel (0.0 = unknown).
        crs:                   EPSG CRS string (empty = no CRS / non-georeferenced).
        bounds_wgs84:          [min_lon, min_lat, max_lon, max_lat] (empty = unknown).
        cloud_cover_pct:       Scene cloud fraction 0–100 (-1.0 = N/A).
        is_georeferenced:      True if file carries a valid CRS + affine transform.
        file_format:           Driver/extension string (e.g. 'geotiff', 'png').
        metadata:              Freeform GDAL / EXIF / STAC header metadata.

    Returns:
        ImageModalityEntry TypedDict instance.
    """
    if file_format == "" and image_path:
        import os
        ext = os.path.splitext(image_path.split("?")[0].lower())[1].lstrip(".")
        file_format = ext or "unknown"

    return ImageModalityEntry(
        image_id=str(uuid.uuid4()),
        image_path=image_path,
        detected_modality=detected_modality,
        file_format=file_format,
        band_count=band_count,
        band_names=band_names or [],
        spatial_role=spatial_role,
        sensor_name=sensor_name,
        acquisition_timestamp=acquisition_timestamp,
        resolution_meters=resolution_meters,
        crs=crs,
        bounds_wgs84=bounds_wgs84 or [],
        cloud_cover_pct=cloud_cover_pct,
        is_georeferenced=is_georeferenced,
        metadata=metadata or {}
    )


def make_trace_entry(
    tool_name: str,
    parameters: Dict[str, Any],
    status: str = "success",
    result_summary: str = "",
    confidence: float = 0.0,
    duration_ms: float = 0.0,
    node_name: str = "",
    error: str = "",
    output_keys: Optional[List[str]] = None,
    timestamp_start: str = "",
    timestamp_end: str = ""
) -> ExecutionTraceEntry:
    """
    Construct a fully-populated ExecutionTraceEntry dict.

    Intended to be called immediately after a tool returns so that latency,
    status, and confidence are captured in-place by the calling graph node.

    Args:
        tool_name:       Canonical tool identifier (e.g. 'vqa_tool').
        parameters:      Exact input parameters dict passed to the tool.
        status:          'success' | 'error' | 'skipped' | 'timeout'.
        result_summary:  One-sentence human-readable outcome.
        confidence:      Tool-reported confidence [0.0 – 1.0].
        duration_ms:     Wall-clock execution latency in milliseconds.
        node_name:       LangGraph node that invoked the tool.
        error:           Exception message or traceback if status == 'error'.
        output_keys:     Top-level keys present in the tool's output dict.
        timestamp_start: ISO-8601 UTC invocation time (auto-filled to utcnow if empty).
        timestamp_end:   ISO-8601 UTC return time (auto-filled to utcnow if empty).

    Returns:
        ExecutionTraceEntry TypedDict instance.
    """
    now_iso = datetime.utcnow().isoformat()
    return ExecutionTraceEntry(
        trace_id=str(uuid.uuid4()),
        tool_name=tool_name,
        node_name=node_name,
        parameters=parameters,
        status=status,
        result_summary=result_summary,
        confidence=confidence,
        duration_ms=duration_ms,
        timestamp_start=timestamp_start or now_iso,
        timestamp_end=timestamp_end or now_iso,
        error=error,
        output_keys=output_keys or []
    )


def make_empty_rs_state(
    raw_query: str,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> RSAgentState:
    """
    Create a blank RSAgentState dict ready to be injected into a LangGraph run.

    All list fields are initialised to empty lists and all optional scalars
    to None so the graph's reducer annotations start from a clean baseline.

    Args:
        raw_query:   Exact user prompt (required).
        request_id:  UUID for this request (auto-generated if not supplied).
        user_id:     Authenticated user ID (None for anonymous requests).
        session_id:  Browser / API session identifier.

    Returns:
        Fully initialised RSAgentState dict.
    """
    now_iso = datetime.utcnow().isoformat()
    empty_tool_outputs: IntermediateToolOutputs = IntermediateToolOutputs(
        vqa_answer="",
        vqa_confidence=0.0,
        vqa_embedding_shape=[],
        bounding_boxes=[],
        spatial_masks=[],
        land_cover_labels={},
        land_cover_confidence=0.0,
        fusion_result={},
        fusion_confidence=0.0,
        damage_assessment={},
        general_answer="",
        raw_tool_outputs={}
    )
    return RSAgentState(
        # Identity
        request_id=request_id or str(uuid.uuid4()),
        user_id=user_id,
        session_id=session_id,
        # Query
        raw_query=raw_query,
        query=raw_query,
        geo_context=None,
        # Image inputs
        image_inputs=[],
        # Task classification
        classified_task=None,
        task_classification_confidence=None,
        task_reasoning=None,
        thought_trace=[],
        plan_steps=None,
        current_step=0,
        # Specialist selection
        selected_specialist_models=[],
        execution_queue=[],
        completed_specialists=[],
        is_compound_task=False,
        specialist_model_configs={},
        # Validation
        validation_flags={},
        is_valid=True,
        validation_errors=[],
        validation_warnings=[],
        requires_clarification=False,
        evidence_inconclusive=False,
        evidence_retry_count=0,
        # Message bus
        messages=[],
        active_agent="orchestrator",
        routing_history=[],
        # Execution trace
        execution_trace=[],
        tool_logs=[],
        # Intermediate outputs
        tool_outputs=empty_tool_outputs,
        intermediate_outputs={},
        tool_confidence_scores={},
        # Final outputs
        final_response=None,
        executive_summary=None,
        detailed_analysis=None,
        spatial_outputs={},
        bounding_boxes=[],
        change_mask=None,
        confidence_score=None,
        confidence_breakdown=None,
        artifacts=[],
        # Lifecycle
        status="pending",
        error=None,
        created_at=now_iso,
        updated_at=now_iso
    )


# ---------------------------------------------------------------------------
# Pydantic State Model (for FastAPI Request/Response & Database Serialization)
# ---------------------------------------------------------------------------

class AgentStateModel(BaseModel):
    """
    Comprehensive Pydantic model representation of the agent state.
    Used for request validation, REST API payloads, and PostgreSQL persistence.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Inputs
    raw_query: str = Field(..., description="Exact raw user query as received")
    query: Optional[str] = Field(None, description="Processed or expanded user query")
    geo_context: GeoSpatialContext = Field(default_factory=GeoSpatialContext)
    modalities: ModalityInputs = Field(default_factory=ModalityInputs)
    
    # Thought Process & Classification
    classified_task: Optional[Union[TaskType, str]] = Field(None, description="Classified task: 'VQA', 'change_detection', 'grounding', etc.")
    task_classification_confidence: Optional[float] = Field(None, description="Confidence score of task classifier (0.0 to 1.0)")
    task_reasoning: Optional[str] = Field(None, description="Chain-of-thought rationale for task classification")
    thought_trace: List[ReasoningStep] = Field(default_factory=list, description="Reasoning and thought trajectory log")
    plan_steps: List[str] = Field(default_factory=list)
    current_step: int = 0
    
    # Multi-Step Execution Queue
    selected_specialist_models: List[Union[SpecialistModelType, str]] = Field(
        default_factory=list, 
        description="Specialist models selected for execution: e.g. ['cross_modal_fusion', 'change_detector']"
    )
    execution_queue: List[str] = Field(default_factory=list)
    completed_specialists: List[str] = Field(default_factory=list)
    is_compound_task: bool = False
    specialist_model_configs: Dict[str, Any] = Field(default_factory=dict, description="Inference parameters per model")
    
    # Validation Flags
    validation_flags: ValidationFlags = Field(default_factory=ValidationFlags)
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    requires_clarification: bool = False
    evidence_inconclusive: bool = False
    evidence_retry_count: int = 0
    
    # Trajectory & Messages
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    active_agent: str = "orchestrator"
    routing_history: List[str] = Field(default_factory=list)
    
    # Tool & Model Outputs
    tool_logs: List[ToolExecutionLog] = Field(default_factory=list)
    intermediate_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Output Tracking: Textual Answers, Spatial Outputs & Confidence Scores
    final_response: Optional[str] = Field(None, description="Synthesized textual answer for the user")
    executive_summary: Optional[str] = Field(None, description="Executive summary for dashboard preview")
    detailed_analysis: Optional[str] = Field(None, description="In-depth analytical breakdown")
    
    spatial_outputs: SpatialOutputs = Field(default_factory=SpatialOutputs, description="Container for all spatial outputs")
    bounding_boxes: List[BoundingBoxOutput] = Field(default_factory=list, description="Grounding bounding boxes")
    change_mask: Optional[ChangeMaskOutput] = Field(None, description="Bi-temporal change detection mask")
    
    confidence_score: Optional[float] = Field(None, description="Master confidence score (0.0 to 1.0)")
    confidence_breakdown: ConfidenceScores = Field(default_factory=ConfidenceScores, description="Confidence breakdown per module")
    
    artifacts: List[Artifact] = Field(default_factory=list, description="Output artifacts (masks, overlays, GeoJSON)")
    
    # Lifecycle Status
    status: RequestStatus = RequestStatus.PENDING
    error: Optional[str] = None
    
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @staticmethod
    def _to_dict_deep(obj: Any, depth: int = 0) -> Any:
        """Deep dictionary converter supporting Pydantic models, dicts, enums, and lists with recursion guard."""
        if depth > 15:
            return str(obj)
        if obj is None:
            return None
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, list):
            return [AgentStateModel._to_dict_deep(item, depth + 1) for item in obj]
        if isinstance(obj, dict):
            return {str(k): AgentStateModel._to_dict_deep(v, depth + 1) for k, v in obj.items()}
        if hasattr(obj, 'model_dump') and callable(obj.model_dump):
            try:
                raw = obj.model_dump()
                return AgentStateModel._to_dict_deep(raw, depth + 1)
            except Exception:
                return str(obj)
        if hasattr(obj, 'dict') and callable(obj.dict):
            try:
                raw = obj.dict()
                return AgentStateModel._to_dict_deep(raw, depth + 1)
            except Exception:
                return str(obj)
        if hasattr(obj, '__dict__'):
            try:
                return {str(k): AgentStateModel._to_dict_deep(v, depth + 1) for k, v in obj.__dict__.items() if not str(k).startswith('_')}
            except Exception:
                return str(obj)
        return str(obj)

    def to_graph_state(self) -> AgentState:
        """Convert Pydantic model into a LangGraph-compatible TypedDict state."""
        geo_dict = self._to_dict_deep(self.geo_context)
        mod_dict = self._to_dict_deep(self.modalities)
        
        single_img = self._to_dict_deep(self.modalities.single_image) if self.modalities else None
        bi_temporal = self._to_dict_deep(self.modalities.bi_temporal_pair) if self.modalities else None
        opt_sar = self._to_dict_deep(self.modalities.optical_sar_pair) if self.modalities else None
        uploaded = self._to_dict_deep(self.modalities.uploaded_images) if self.modalities else []
        
        val_flags_dict = self._to_dict_deep(self.validation_flags)
        
        task_val = self.classified_task.value if hasattr(self.classified_task, 'value') else self.classified_task
        models_val = [m.value if hasattr(m, 'value') else str(m) for m in self.selected_specialist_models]
        
        # Spatial outputs handling
        sp_outputs_dict = self._to_dict_deep(self.spatial_outputs)
        bboxes_list = [self._to_dict_deep(b) for b in (self.bounding_boxes or (self.spatial_outputs.bounding_boxes if self.spatial_outputs else []))]
        c_mask_dict = self._to_dict_deep(self.change_mask or (self.spatial_outputs.change_mask if self.spatial_outputs else None))
        
        # Confidence score handling
        conf_score = self.confidence_score if self.confidence_score is not None else (self.confidence_breakdown.overall if self.confidence_breakdown else None)
        conf_breakdown_dict = self._to_dict_deep(self.confidence_breakdown)
        
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "raw_query": self.raw_query,
            "query": self.query or self.raw_query,
            "geo_context": geo_dict,
            "single_image": single_img,
            "bi_temporal_pair": bi_temporal,
            "optical_sar_pair": opt_sar,
            "uploaded_images": uploaded,
            "modalities": mod_dict,
            "classified_task": task_val,
            "task_classification_confidence": self.task_classification_confidence,
            "task_reasoning": self.task_reasoning,
            "thought_trace": [self._to_dict_deep(t) for t in self.thought_trace],
            "plan_steps": self.plan_steps,
            "current_step": self.current_step,
            "selected_specialist_models": models_val,
            "execution_queue": self.execution_queue or list(models_val),
            "completed_specialists": self.completed_specialists,
            "is_compound_task": self.is_compound_task or (len(models_val) > 1),
            "specialist_model_configs": self.specialist_model_configs,
            "validation_flags": val_flags_dict,
            "is_valid": self.is_valid and (self.validation_flags.is_valid if hasattr(self.validation_flags, 'is_valid') else True),
            "validation_errors": self.validation_errors + (self.validation_flags.validation_errors if hasattr(self.validation_flags, 'validation_errors') else []),
            "validation_warnings": self.validation_warnings + (self.validation_flags.validation_warnings if hasattr(self.validation_flags, 'validation_warnings') else []),
            "requires_clarification": self.requires_clarification or (self.validation_flags.requires_human_clarification if hasattr(self.validation_flags, 'requires_human_clarification') else False),
            "evidence_inconclusive": self.evidence_inconclusive or (self.validation_flags.evidence_inconclusive if hasattr(self.validation_flags, 'evidence_inconclusive') else False),
            "evidence_retry_count": self.evidence_retry_count,
            "messages": self.messages,
            "active_agent": self.active_agent,
            "routing_history": self.routing_history,
            "tool_logs": [self._to_dict_deep(log) for log in self.tool_logs],
            "intermediate_outputs": self.intermediate_outputs,
            
            # Outputs
            "final_response": self.final_response,
            "executive_summary": self.executive_summary,
            "detailed_analysis": self.detailed_analysis,
            "spatial_outputs": sp_outputs_dict,
            "bounding_boxes": bboxes_list,
            "change_mask": c_mask_dict,
            "confidence_score": conf_score,
            "confidence_breakdown": conf_breakdown_dict,
            "artifacts": [self._to_dict_deep(art) for art in self.artifacts],
            
            # Lifecycle Status
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "error": self.error
        }
