from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jpype
import numpy as np
import orekit_jpype
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
TLE_FILE = PROJECT_ROOT / "tle.tle"
OUTPUT_DIR = PROJECT_ROOT / "output"
OREKIT_DATA_ZIP = PROJECT_ROOT / "orekit-data.zip"
OREKIT_DATA_URL = "https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip"
ALLOWED_CONSTELLATIONS = ("IRIDIUM", "ORBCOMM")


@dataclass(frozen=True)
class ScenarioConfig:
    start_time_utc: datetime = datetime(2026, 3, 24, 7, 0, 0, tzinfo=timezone.utc)
    coarse_step_sec: float = 1.0
    fine_step_sec: float = 0.01
    min_elevation_deg: float = 10.0
    min_pass_duration_sec: float = 3.0
    initial_visibility_window_sec: float = 1800.0
    max_search_hours: float = 12.0
    target_pass_count: int = 2


@dataclass(frozen=True)
class ReceiverConfig:
    latitude_deg: float = 45.772625
    longitude_deg: float = 126.682625
    altitude_m: float = 154.0


@dataclass(frozen=True)
class HpopConfig:
    mass_kg: float = 260.0
    drag_area_m2: float = 1.5
    drag_coefficient: float = 2.2
    gravity_degree: int = 70
    gravity_order: int = 70
    position_tolerance_m: float = 10.0
    min_step_sec: float = 1.0e-3
    max_step_sec: float = 300.0


@dataclass(frozen=True)
class PassWindow:
    start_time_utc: datetime
    end_time_utc: datetime


SCENARIO = ScenarioConfig()
RECEIVER = ReceiverConfig()
HPOP = HpopConfig()


def download_file(url: str, destination: Path, max_retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted entry found while validating orekit-data.zip.")
            return
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if destination.exists():
                destination.unlink()
    raise RuntimeError(f"Failed to download Orekit data from {url}") from last_error


def ensure_orekit_ready(orekit_data_zip: Path) -> None:
    if not orekit_data_zip.exists():
        download_file(OREKIT_DATA_URL, orekit_data_zip)
    else:
        try:
            with zipfile.ZipFile(orekit_data_zip) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted archive entry detected.")
        except zipfile.BadZipFile:
            download_file(OREKIT_DATA_URL, orekit_data_zip)

    if not jpype.isJVMStarted():
        orekit_jpype.initVM()

    from orekit_jpype.pyhelpers import setup_orekit_curdir

    setup_orekit_curdir(str(orekit_data_zip))


def absolute_date_helpers() -> tuple[Any, Any]:
    from orekit_jpype.pyhelpers import absolutedate_to_datetime, datetime_to_absolutedate

    return absolutedate_to_datetime, datetime_to_absolutedate


def read_tle_catalog(tle_path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip("\n") for line in tle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"TLE file {tle_path} does not contain complete 3-line sets.")

    catalog: list[dict[str, str]] = []
    for index in range(0, len(lines), 3):
        name = lines[index].strip()
        constellation = classify_constellation(name)
        if constellation is None:
            continue
        catalog.append(
            {
                "name": name,
                "line1": lines[index + 1].strip(),
                "line2": lines[index + 2].strip(),
                "constellation": constellation,
            }
        )
    return catalog


def classify_constellation(name: str) -> str | None:
    normalized = name.upper()
    for constellation in ALLOWED_CONSTELLATIONS:
        if normalized.startswith(constellation):
            return constellation
    return None


def make_datetime_grid(start_time: datetime, end_time: datetime, step_sec: float) -> list[datetime]:
    total_seconds = (end_time - start_time).total_seconds()
    sample_count = int(round(total_seconds / step_sec)) + 1
    return [start_time + timedelta(seconds=index * step_sec) for index in range(sample_count)]


def vector3_to_numpy(vector: Any) -> np.ndarray:
    return np.array([vector.getX(), vector.getY(), vector.getZ()], dtype=float)


def pv_to_numpy(pv_coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    return vector3_to_numpy(pv_coordinates.getPosition()), vector3_to_numpy(pv_coordinates.getVelocity())


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def find_pass_windows(visible_mask: np.ndarray, coarse_step_sec: float, min_duration_sec: float) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    in_window = False
    start_idx = 0

    for idx, is_visible in enumerate(visible_mask):
        if is_visible and not in_window:
            in_window = True
            start_idx = idx
        elif not is_visible and in_window:
            end_idx = idx - 1
            duration = (end_idx - start_idx) * coarse_step_sec
            if duration >= min_duration_sec:
                windows.append((start_idx, end_idx))
            in_window = False

    if in_window:
        end_idx = len(visible_mask) - 1
        duration = (end_idx - start_idx) * coarse_step_sec
        if duration >= min_duration_sec:
            windows.append((start_idx, end_idx))

    return windows


def create_frames(receiver: ReceiverConfig) -> dict[str, Any]:
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.utils import Constants, IERSConventions

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    eme2000 = FramesFactory.getEME2000()
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )
    site = GeodeticPoint(
        math.radians(receiver.latitude_deg),
        math.radians(receiver.longitude_deg),
        receiver.altitude_m,
    )
    station = TopocentricFrame(earth, site, "receiver")
    receiver_ecef = vector3_to_numpy(earth.transform(site))
    return {
        "earth_fixed": itrf,
        "inertial": eme2000,
        "earth": earth,
        "station": station,
        "receiver_ecef_m": receiver_ecef,
    }


def create_tle(name: str, line1: str, line2: str) -> Any:
    from org.orekit.propagation.analytical.tle import TLE

    _ = name
    return TLE(line1, line2)


def create_sgp4_propagator(tle: Any) -> Any:
    from org.orekit.propagation.analytical.tle import TLEPropagator

    return TLEPropagator.selectExtrapolator(tle)


def create_hpop_propagator(initial_state: Any, earth_fixed_frame: Any, config: HpopConfig) -> Any:
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.models.earth.atmosphere import HarrisPriester
    from org.orekit.orbits import OrbitType
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.utils import Constants

    orbit = initial_state.getOrbit()
    tolerances = NumericalPropagator.tolerances(config.position_tolerance_m, orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(config.min_step_sec, config.max_step_sec, tolerances[0], tolerances[1])
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)

    gravity_provider = GravityFieldFactory.getNormalizedProvider(config.gravity_degree, config.gravity_order)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(earth_fixed_frame, gravity_provider))

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_fixed_frame,
    )
    atmosphere = HarrisPriester(CelestialBodyFactory.getSun(), earth)
    spacecraft = IsotropicDrag(config.drag_area_m2, config.drag_coefficient)
    propagator.addForceModel(DragForce(atmosphere, spacecraft))
    return propagator


def compute_elevation_deg(propagator: Any, date: Any, frames: dict[str, Any]) -> float:
    pv_ecef = propagator.getPVCoordinates(date, frames["earth_fixed"])
    return math.degrees(frames["station"].getElevation(pv_ecef.getPosition(), frames["earth_fixed"], date))


def refine_pass_boundary(
    propagator: Any,
    frames: dict[str, Any],
    datetime_to_absolutedate: Any,
    lower_dt: datetime,
    upper_dt: datetime,
    threshold_deg: float,
    is_rising_edge: bool,
) -> datetime:
    if lower_dt >= upper_dt:
        return lower_dt

    tolerance = SCENARIO.fine_step_sec
    low = lower_dt
    high = upper_dt

    while (high - low).total_seconds() > tolerance:
        midpoint = low + (high - low) / 2
        elevation_mid = compute_elevation_deg(propagator, datetime_to_absolutedate(midpoint), frames)
        is_visible_mid = elevation_mid >= threshold_deg
        if is_rising_edge:
            if is_visible_mid:
                high = midpoint
            else:
                low = midpoint
        else:
            if is_visible_mid:
                low = midpoint
            else:
                high = midpoint

    return high if is_rising_edge else low


def find_satellite_passes(
    propagator: Any,
    frames: dict[str, Any],
    datetime_to_absolutedate: Any,
    search_start_dt: datetime,
    search_end_dt: datetime,
    coarse_step_sec: float,
    min_elevation_deg: float,
    min_duration_sec: float,
) -> list[PassWindow]:
    coarse_datetimes = make_datetime_grid(search_start_dt, search_end_dt, coarse_step_sec)
    coarse_dates = [datetime_to_absolutedate(value) for value in coarse_datetimes]
    visible_mask = np.zeros(len(coarse_dates), dtype=bool)

    for index, date in enumerate(coarse_dates):
        visible_mask[index] = compute_elevation_deg(propagator, date, frames) >= min_elevation_deg

    windows: list[PassWindow] = []
    for start_idx, end_idx in find_pass_windows(visible_mask, coarse_step_sec, min_duration_sec):
        start_dt = coarse_datetimes[start_idx]
        end_dt = coarse_datetimes[end_idx]

        if start_idx > 0:
            start_dt = refine_pass_boundary(
                propagator,
                frames,
                datetime_to_absolutedate,
                coarse_datetimes[start_idx - 1],
                coarse_datetimes[start_idx],
                min_elevation_deg,
                is_rising_edge=True,
            )

        if end_idx < len(coarse_datetimes) - 1:
            end_dt = refine_pass_boundary(
                propagator,
                frames,
                datetime_to_absolutedate,
                coarse_datetimes[end_idx],
                coarse_datetimes[end_idx + 1],
                min_elevation_deg,
                is_rising_edge=False,
            )

        windows.append(PassWindow(start_time_utc=start_dt, end_time_utc=end_dt))

    return windows


def compute_range_and_range_rate(
    receiver_ecef_m: np.ndarray,
    satellite_ecef_pos_m: np.ndarray,
    satellite_ecef_vel_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    line_of_sight = satellite_ecef_pos_m - receiver_ecef_m[None, :]
    geometric_range_m = np.linalg.norm(line_of_sight, axis=1)
    line_of_sight_unit = line_of_sight / geometric_range_m[:, None]
    geometric_range_rate_mps = np.sum(satellite_ecef_vel_mps * line_of_sight_unit, axis=1)
    return geometric_range_m, geometric_range_rate_mps


def build_rtn_frame(
    reference_pos_eci_m: np.ndarray,
    reference_vel_eci_mps: np.ndarray,
) -> np.ndarray:
    radial = reference_pos_eci_m / np.linalg.norm(reference_pos_eci_m)
    normal = np.cross(reference_pos_eci_m, reference_vel_eci_mps)
    normal = normal / np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    transverse = transverse / np.linalg.norm(transverse)
    return np.column_stack((radial, transverse, normal))


def compute_rtn_residual_series(
    reference_pos_eci_m: np.ndarray,
    reference_vel_eci_mps: np.ndarray,
    target_pos_eci_m: np.ndarray,
    target_vel_eci_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = reference_pos_eci_m.shape[0]
    rtn_frames = np.zeros((sample_count, 3, 3), dtype=float)
    position_residual_rtn_m = np.zeros((sample_count, 3), dtype=float)
    velocity_residual_rtn_mps = np.zeros((sample_count, 3), dtype=float)

    for index in range(sample_count):
        rtn_frame = build_rtn_frame(reference_pos_eci_m[index], reference_vel_eci_mps[index])
        rtn_frames[index] = rtn_frame
        position_residual_rtn_m[index] = rtn_frame.T @ (target_pos_eci_m[index] - reference_pos_eci_m[index])
        velocity_residual_rtn_mps[index] = rtn_frame.T @ (target_vel_eci_mps[index] - reference_vel_eci_mps[index])

    return rtn_frames, position_residual_rtn_m, velocity_residual_rtn_mps


def build_observation_mask(time_seconds: np.ndarray, pass_windows_time_sec: np.ndarray) -> np.ndarray:
    observation_mask = np.zeros(time_seconds.shape, dtype=bool)
    for window_start_sec, window_end_sec in pass_windows_time_sec:
        observation_mask |= (time_seconds >= window_start_sec) & (time_seconds <= window_end_sec)
    return observation_mask


def overlaps_time_window(pass_window: PassWindow, window_start_dt: datetime, window_end_dt: datetime) -> bool:
    return pass_window.end_time_utc >= window_start_dt and pass_window.start_time_utc <= window_end_dt


def build_segment_phase_masks(
    time_seconds: np.ndarray,
    pass_windows_time_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pass_windows_time_sec.shape[0] < 2:
        raise ValueError("At least two pass windows are required to build phase masks.")

    first_pass_start_sec, first_pass_end_sec = pass_windows_time_sec[0]
    second_pass_start_sec, second_pass_end_sec = pass_windows_time_sec[1]

    first_pass_mask = (time_seconds >= first_pass_start_sec) & (time_seconds <= first_pass_end_sec)
    gap_mask = (time_seconds > first_pass_end_sec) & (time_seconds < second_pass_start_sec)
    second_pass_mask = (time_seconds >= second_pass_start_sec) & (time_seconds <= second_pass_end_sec)
    return first_pass_mask, gap_mask, second_pass_mask


def generate_satellite_data_files(max_satellites: int | None = None) -> list[dict[str, Any]]:
    ensure_orekit_ready(OREKIT_DATA_ZIP)
    absolutedate_to_datetime, datetime_to_absolutedate = absolute_date_helpers()
    frames = create_frames(RECEIVER)
    output_records: list[dict[str, Any]] = []

    search_start_dt = SCENARIO.start_time_utc
    search_end_dt = search_start_dt + timedelta(hours=SCENARIO.max_search_hours)
    initial_visibility_end_dt = search_start_dt + timedelta(seconds=SCENARIO.initial_visibility_window_sec)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tle_catalog = read_tle_catalog(TLE_FILE)
    if max_satellites is not None:
        tle_catalog = tle_catalog[:max_satellites]
    print(f"Loaded {len(tle_catalog)} supported TLE records from {TLE_FILE}")

    from org.orekit.orbits import CartesianOrbit
    from org.orekit.propagation import SpacecraftState
    from org.orekit.utils import Constants

    for sat_idx, tle_entry in enumerate(tle_catalog, start=1):
        tle = create_tle(tle_entry["name"], tle_entry["line1"], tle_entry["line2"])
        sgp4 = create_sgp4_propagator(tle)

        pass_windows = find_satellite_passes(
            sgp4,
            frames,
            datetime_to_absolutedate,
            search_start_dt,
            search_end_dt,
            SCENARIO.coarse_step_sec,
            SCENARIO.min_elevation_deg,
            SCENARIO.min_pass_duration_sec,
        )

        qualifying_pass_index = next(
            (
                index
                for index, pass_window in enumerate(pass_windows)
                if overlaps_time_window(pass_window, search_start_dt, initial_visibility_end_dt)
            ),
            None,
        )

        if qualifying_pass_index is None:
            print(
                f"[{sat_idx:02d}/{len(tle_catalog):02d}] {tle_entry['name']}: "
                "no pass found within the initial 30-minute visibility window"
            )
            continue

        if qualifying_pass_index + SCENARIO.target_pass_count > len(pass_windows):
            print(
                f"[{sat_idx:02d}/{len(tle_catalog):02d}] {tle_entry['name']}: "
                "found an initial-window pass, but not enough subsequent passes for export"
            )
            continue

        selected_passes = pass_windows[
            qualifying_pass_index : qualifying_pass_index + SCENARIO.target_pass_count
        ]
        segment_start_dt = selected_passes[0].start_time_utc
        segment_end_dt = selected_passes[-1].end_time_utc

        tle_epoch = tle.getDate()
        tle_epoch_dt = absolutedate_to_datetime(tle_epoch, tz_aware=True)
        tle_age_hours = max((segment_start_dt - tle_epoch_dt).total_seconds() / 3600.0, 0.0)

        segment_start_date = datetime_to_absolutedate(segment_start_dt)
        segment_end_date = datetime_to_absolutedate(segment_end_dt)

        initial_pv = sgp4.getPVCoordinates(tle_epoch, frames["inertial"])
        initial_orbit = CartesianOrbit(initial_pv, frames["inertial"], tle_epoch, Constants.WGS84_EARTH_MU)
        initial_state = SpacecraftState(initial_orbit, HPOP.mass_kg)
        hpop_bridge = create_hpop_propagator(initial_state, frames["earth_fixed"], HPOP)
        segment_start_state = hpop_bridge.propagate(segment_start_date)

        segment_hpop = create_hpop_propagator(segment_start_state, frames["earth_fixed"], HPOP)
        ephemeris_generator = segment_hpop.getEphemerisGenerator()
        segment_hpop.propagate(segment_end_date)
        hpop_ephemeris = ephemeris_generator.getGeneratedEphemeris()

        fine_datetimes = make_datetime_grid(segment_start_dt, segment_end_dt, SCENARIO.fine_step_sec)
        fine_dates = [datetime_to_absolutedate(value) for value in fine_datetimes]
        time_seconds = np.array(
            [(value - segment_start_dt).total_seconds() for value in fine_datetimes],
            dtype=float,
        )

        sample_count = len(fine_dates)
        hpop_eci_pos = np.zeros((sample_count, 3), dtype=float)
        hpop_eci_vel = np.zeros((sample_count, 3), dtype=float)
        hpop_ecef_pos = np.zeros((sample_count, 3), dtype=float)
        hpop_ecef_vel = np.zeros((sample_count, 3), dtype=float)
        sgp4_eci_pos = np.zeros((sample_count, 3), dtype=float)
        sgp4_eci_vel = np.zeros((sample_count, 3), dtype=float)
        sgp4_ecef_pos = np.zeros((sample_count, 3), dtype=float)
        sgp4_ecef_vel = np.zeros((sample_count, 3), dtype=float)

        for index, date in enumerate(fine_dates):
            hpop_state = hpop_ephemeris.propagate(date)
            hpop_eci_pv = hpop_state.getPVCoordinates(frames["inertial"])
            hpop_ecef_pv = hpop_state.getPVCoordinates(frames["earth_fixed"])
            sgp4_eci_pv = sgp4.getPVCoordinates(date, frames["inertial"])
            sgp4_ecef_pv = sgp4.getPVCoordinates(date, frames["earth_fixed"])

            hpop_eci_pos[index], hpop_eci_vel[index] = pv_to_numpy(hpop_eci_pv)
            hpop_ecef_pos[index], hpop_ecef_vel[index] = pv_to_numpy(hpop_ecef_pv)
            sgp4_eci_pos[index], sgp4_eci_vel[index] = pv_to_numpy(sgp4_eci_pv)
            sgp4_ecef_pos[index], sgp4_ecef_vel[index] = pv_to_numpy(sgp4_ecef_pv)

        pass_windows_time_sec = np.array(
            [
                [
                    (window.start_time_utc - segment_start_dt).total_seconds(),
                    (window.end_time_utc - segment_start_dt).total_seconds(),
                ]
                for window in selected_passes
            ],
            dtype=float,
        )
        observation_mask = build_observation_mask(time_seconds, pass_windows_time_sec)
        first_pass_mask, prediction_gap_mask, second_pass_mask = build_segment_phase_masks(
            time_seconds,
            pass_windows_time_sec,
        )
        pseudorange_truth_m, pseudorange_rate_truth_mps = compute_range_and_range_rate(
            frames["receiver_ecef_m"],
            hpop_ecef_pos,
            hpop_ecef_vel,
        )
        rtn_frames_eci_to_rtn, sgp4_to_hpop_pos_rtn_m, sgp4_to_hpop_vel_rtn_mps = compute_rtn_residual_series(
            sgp4_eci_pos,
            sgp4_eci_vel,
            hpop_eci_pos,
            hpop_eci_vel,
        )

        pseudorange_m = np.full(sample_count, np.nan, dtype=float)
        pseudorange_rate_mps = np.full(sample_count, np.nan, dtype=float)
        pseudorange_m[observation_mask] = pseudorange_truth_m[observation_mask]
        pseudorange_rate_mps[observation_mask] = pseudorange_rate_truth_mps[observation_mask]

        file_stem = safe_name(tle_entry["name"])
        output_file = OUTPUT_DIR / f"{file_stem}.npz"
        np.savez_compressed(
            output_file,
            satellite_name=np.array(tle_entry["name"]),
            constellation=np.array(tle_entry["constellation"]),
            catalog_id=np.array(int(tle.getSatelliteNumber())),
            tle_line1=np.array(tle_entry["line1"]),
            tle_line2=np.array(tle_entry["line2"]),
            tle_epoch_utc=np.array(tle_epoch_dt.isoformat()),
            tle_age_hours_at_segment_start=np.array(tle_age_hours),
            search_anchor_utc=np.array(SCENARIO.start_time_utc.isoformat()),
            initial_visibility_window_end_utc=np.array(initial_visibility_end_dt.isoformat()),
            segment_start_utc=np.array(segment_start_dt.isoformat()),
            segment_end_utc=np.array(segment_end_dt.isoformat()),
            receiver_lla_deg_m=np.array(
                [RECEIVER.latitude_deg, RECEIVER.longitude_deg, RECEIVER.altitude_m],
                dtype=float,
            ),
            receiver_ecef_m=frames["receiver_ecef_m"],
            time_seconds=time_seconds,
            pass_windows_time_sec=pass_windows_time_sec,
            pass_windows_utc_iso=np.array(
                [[window.start_time_utc.isoformat(), window.end_time_utc.isoformat()] for window in selected_passes]
            ),
            observation_valid=observation_mask,
            first_pass_mask=first_pass_mask,
            prediction_gap_mask=prediction_gap_mask,
            second_pass_mask=second_pass_mask,
            pseudorange_m=pseudorange_m,
            pseudorange_rate_mps=pseudorange_rate_mps,
            hpop_eci_pos_m=hpop_eci_pos,
            hpop_eci_vel_mps=hpop_eci_vel,
            hpop_ecef_pos_m=hpop_ecef_pos,
            hpop_ecef_vel_mps=hpop_ecef_vel,
            tracked_eci_pos_m=hpop_eci_pos,
            tracked_eci_vel_mps=hpop_eci_vel,
            tracked_ecef_pos_m=hpop_ecef_pos,
            tracked_ecef_vel_mps=hpop_ecef_vel,
            sgp4_eci_pos_m=sgp4_eci_pos,
            sgp4_eci_vel_mps=sgp4_eci_vel,
            sgp4_ecef_pos_m=sgp4_ecef_pos,
            sgp4_ecef_vel_mps=sgp4_ecef_vel,
            rtn_frame_eci_to_rtn=rtn_frames_eci_to_rtn,
            sgp4_to_hpop_pos_rtn_m=sgp4_to_hpop_pos_rtn_m,
            sgp4_to_hpop_vel_rtn_mps=sgp4_to_hpop_vel_rtn_mps,
        )

        record = {
            "satellite_name": tle_entry["name"],
            "constellation": tle_entry["constellation"],
            "catalog_id": int(tle.getSatelliteNumber()),
            "segment_start_utc": segment_start_dt.isoformat(),
            "segment_end_utc": segment_end_dt.isoformat(),
            "pass_windows_utc": [
                [window.start_time_utc.isoformat(), window.end_time_utc.isoformat()] for window in selected_passes
            ],
            "sample_count": sample_count,
            "file": str(output_file.resolve()),
        }
        output_records.append(record)
        print(
            f"[{sat_idx:02d}/{len(tle_catalog):02d}] {tle_entry['name']}: "
            f"saved {sample_count} samples -> {output_file.name}"
        )

    summary_file = OUTPUT_DIR / "satellite_data_index.json"
    summary = {
        "scenario": asdict(SCENARIO),
        "receiver": asdict(RECEIVER),
        "hpop": asdict(HPOP),
        "tle_file": str(TLE_FILE.resolve()),
        "orekit_data_zip": str(OREKIT_DATA_ZIP.resolve()),
        "satellites": output_records,
    }
    summary_file.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote summary index -> {summary_file}")
    return output_records


def main() -> int:
    try:
        records = generate_satellite_data_files()
        if not records:
            print("No satellite data files were generated.")
        return 0
    except Exception as exc:  # pragma: no cover - command line entry point
        print(f"Data generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
