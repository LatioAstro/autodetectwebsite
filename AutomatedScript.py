import argparse
import importlib
import html
import json
import os
import shutil
import smtplib
import sys
from datetime import date, datetime, timezone
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defining all paths required for the script.
# The paths can be adjusted and changed if you are running this on another machine.
ROOT = Path(__file__).resolve().parent
PYTHON_FILES = ROOT / 'PythonFiles'
# Downloaded LightCurves and outputs will be stored in this directory.
OUTPUT_DIR = ROOT / 'DownloadedLC'
INCREMENTAL_OUTPUT_DIR = OUTPUT_DIR / 'incremental'
SOURCES_DIR = ROOT / 'Sources'
WEBSITE_SOURCE_DATA_DIR = ROOT / "Sources"
INCREMENTAL_WEEKLY_SUMMARY_PATH = INCREMENTAL_OUTPUT_DIR / 'weekly_incremental_summary.csv'

sys.path.insert(0, str(PYTHON_FILES))

notebook_pipeline = importlib.import_module('notebook_pipeline')

# We check to see what the defined background rate is in the env file, and use the appropriate scaling factor
# to ensure consistent background assumptions in incremental processing.

COSI_BACKGROUND_RATE = float(os.environ.get('COSI_BACKGROUND_RATE', '0.25'))
INSTRUMENT = os.environ.get('INSTRUMENT', 'Fermi-LAT')
ENERGY_RANGE = {'min': 100, 'max': 100000, 'unit': 'MeV'} # Fermi energy range
#INSTRUMENT = os.environ.get('INSTRUMENT', 'COSI')
#ENERGY_RANGE = {'min': 0.2, 'max': 5, 'unit': 'MeV'}   # COSI energy range



# ARM CUTS: 
# 20 ct/s: 1 factor
# 10 ct/s: 1.27 factor
# 1 ct/s: 3.22 factor

if COSI_BACKGROUND_RATE == 20*0.25:
	ARM_reduction = 1
elif COSI_BACKGROUND_RATE == 10*0.25:
	ARM_reduction = 1.27
elif COSI_BACKGROUND_RATE == 1*0.25:
	ARM_reduction = 3.22
else:
	ARM_reduction = 1

SECONDS_PER_WEEK = 7.0 * 24.0 * 60.0 * 60.0
MDP99_AVERAGE_MU = float(os.environ.get('MDP99_AVERAGE_MU', '0.3'))
PLOT_FIGURE_WIDTH = float(os.environ.get('PLOT_FIGURE_WIDTH', '8.0'))
PLOT_FIGURE_HEIGHT = float(os.environ.get('PLOT_FIGURE_HEIGHT', '3.2'))
PLOT_SAVE_DPI = int(os.environ.get('PLOT_SAVE_DPI', '70'))

# Band that a Fermi-LAT flux is converted to when Int_flux_ratio is applied.
COSI_SCALED_ENERGY_RANGE = {'min': 0.2, 'max': 5, 'unit': 'MeV'}


def resolve_flux_view(factor_row: pd.Series | None) -> dict:
	"""
	Decide once per source which flux is plotted and whether a COSI scale applies.
	view is 'COSI' (native COSI data), 'COSI-scaled' (Fermi-LAT data times Int_flux_ratio), or 'Fermi-LAT' (unscaled).
	"""
	if INSTRUMENT == 'COSI':
		# Native COSI data is already in COSI units and must never be rescaled.
		return {'flux_scale': 1.0, 'view': 'COSI'}

	scale = np.nan
	if factor_row is not None and 'Int_flux_ratio' in factor_row.index:
		try:
			scale = float(factor_row['Int_flux_ratio'])
		except (TypeError, ValueError):
			pass
	if np.isfinite(scale) and scale > 0:
		return {'flux_scale': scale, 'view': 'COSI-scaled'}
	return {'flux_scale': 1.0, 'view': 'Fermi-LAT'}


def flux_axis_label(view: str) -> str:
	"""Y-axis label for a view returned by resolve_flux_view."""
	scaled_to_cosi = view == 'COSI-scaled'
	band = COSI_SCALED_ENERGY_RANGE if scaled_to_cosi else ENERGY_RANGE
	prefix = 'COSI ' if view in {'COSI', 'COSI-scaled'} else ''
	low, high, unit = band['min'], band['max'], band['unit']
	if unit == 'MeV' and high >= 1000:
		energy = f'{low:g} MeV - {high / 1000:g} GeV'
	else:
		energy = f'{low:g} - {high:g} {unit}'
	return rf'{prefix}Photon Flux (ph cm$^{{-2}}$ s$^{{-1}}$) [{energy}]'


def read_source_names(path: Path) -> list[str]:
	"""
	This is just a function that reads the names from the NameCSV file and returns a list of source names. 
	It also checks that the 'Name' column is present and that there are no missing values.
	"""
	names = pd.read_csv(path)
	if 'Name' not in names.columns:
		raise ValueError(f'Missing Name column in {path}')
	return names['Name'].dropna().astype(str).tolist()


def build_active_intervals(
	active_rows: pd.DataFrame,
	*,
	start_column: str,
	bin_column: str = 'current_bin_count',
	time_column: str = 'new_point_mjd',
) -> list[tuple[float, float]]:
	if active_rows.empty:
		return []

	intervals = []
	interval_start = active_rows.iloc[0]
	interval_prev = active_rows.iloc[0]
	for _, row in active_rows.iloc[1:].iterrows():
		contiguous = int(row[bin_column]) == int(interval_prev[bin_column]) + 1
		same_start = np.isclose(row[start_column], interval_prev[start_column], atol=1e-6)
		if contiguous and same_start:
			interval_prev = row
			continue
		intervals.append((float(interval_start[start_column]), float(interval_prev[time_column])))
		interval_start = row
		interval_prev = row
	intervals.append((float(interval_start[start_column]), float(interval_prev[time_column])))
	return intervals


def build_interval_mdp_labels(
	active_rows: pd.DataFrame,
	*,
	start_column: str,
	bin_column: str = 'current_bin_count',
	time_column: str = 'new_point_mjd',
) -> list[tuple[float, float, float]]:
	"""
	Build one MDP99 label per active flare interval using the interval-final MDP99 value,
	which represents the full streak up to that flare end.
	"""
	if active_rows.empty:
		return []

	labels: list[tuple[float, float, float]] = []
	interval_start = active_rows.iloc[0]
	interval_prev = active_rows.iloc[0]
	for _, row in active_rows.iloc[1:].iterrows():
		contiguous = int(row[bin_column]) == int(interval_prev[bin_column]) + 1
		same_start = np.isclose(row[start_column], interval_prev[start_column], atol=1e-6)
		if contiguous and same_start:
			interval_prev = row
			continue

		final_mdp99 = float(interval_prev['mdp99_percent']) if np.isfinite(interval_prev['mdp99_percent']) else np.nan
		if np.isfinite(final_mdp99):
			labels.append((float(interval_start[start_column]), float(interval_prev[time_column]), final_mdp99))
		interval_start = row
		interval_prev = row

	final_mdp99 = float(interval_prev['mdp99_percent']) if np.isfinite(interval_prev['mdp99_percent']) else np.nan
	if np.isfinite(final_mdp99):
		labels.append((float(interval_start[start_column]), float(interval_prev[time_column]), final_mdp99))

	return labels


def latest_highlighted_flare_stats(
	result_df: pd.DataFrame,
	active_rows: pd.DataFrame,
	factor_row: pd.Series | None,
	*,
	cosi_background_rate: float,
	arm_reduction: float,
	average_mu: float,
) -> dict[str, float | bool]:
	"""
	Compute metrics for the most recently highlighted flare interval.
	The interval is taken from the latest active flare segment in the same way plotting highlights it.
	"""
	stats: dict[str, float | bool] = {
		'latest_highlighted_mdp99_percent': np.nan,
		'latest_highlighted_mdp99_available': False,
		'latest_highlighted_peak_flux_cosi': np.nan,
		'latest_highlighted_average_flux_cosi': np.nan,
		'latest_highlighted_start_mjd': np.nan,
		'latest_highlighted_end_mjd': np.nan,
	}

	if active_rows.empty:
		return stats

	intervals = build_active_intervals(active_rows, start_column='flare_start_mjd')
	if not intervals:
		return stats

	interval_start, interval_end = intervals[-1]
	stats['latest_highlighted_start_mjd'] = float(interval_start)
	stats['latest_highlighted_end_mjd'] = float(interval_end)

	interval_rows = result_df.loc[
		result_df['new_point_mjd'].between(interval_start, interval_end)
		& result_df['potential_flare_point']
	].copy()
	if interval_rows.empty:
		interval_rows = result_df.loc[result_df['new_point_mjd'].between(interval_start, interval_end)].copy()

	if interval_rows.empty:
		return stats

	interval_rows['new_point_flux'] = pd.to_numeric(interval_rows['new_point_flux'], errors='coerce')
	interval_rows = interval_rows.dropna(subset=['new_point_flux'])
	if interval_rows.empty:
		return stats

	flux_scale = resolve_flux_view(factor_row)['flux_scale']

	stats['latest_highlighted_peak_flux_cosi'] = float(interval_rows['new_point_flux'].max() * flux_scale)
	stats['latest_highlighted_average_flux_cosi'] = float(interval_rows['new_point_flux'].mean() * flux_scale)

	if factor_row is None:
		return stats

	lat_aeff = float(factor_row['Aeff_mean_LAT(cm2)']) if 'Aeff_mean_LAT(cm2)' in factor_row.index else np.nan
	ph_ratio = float(factor_row['ph/s_ratio']) if 'ph/s_ratio' in factor_row.index else np.nan
	if (not np.isfinite(lat_aeff)) or (not np.isfinite(ph_ratio)) or arm_reduction <= 0 or average_mu <= 0:
		return stats

	duration_seconds = float(len(interval_rows) * SECONDS_PER_WEEK)
	mean_flux = float(interval_rows['new_point_flux'].mean())
	background_counts = float(duration_seconds * cosi_background_rate)
	cosi_photon_rate = float(mean_flux * lat_aeff * ph_ratio)
	source_counts = float(cosi_photon_rate * duration_seconds / arm_reduction)
	if source_counts <= 0:
		return stats

	mdp99 = float(notebook_pipeline.compute_mdp99(source_counts, background_counts, average_mu=average_mu))
	if np.isfinite(mdp99):
		stats['latest_highlighted_mdp99_percent'] = mdp99
		stats['latest_highlighted_mdp99_available'] = True

	return stats


def _resolve_plot_column(dataframe: pd.DataFrame, *candidates: str) -> str:
	for candidate in candidates:
		if candidate in dataframe.columns:
			return candidate
	raise KeyError(f'No plotting columns found. Expected one of: {", ".join(candidates)}')


def plot_light_curve(
	dataframe: pd.DataFrame,
	source_name: str,
	output_path: Path,
	title_suffix: str,
	y_label: str | None = None,
	time_range: tuple[float, float] | None = None,
	flare_points: pd.DataFrame | None = None,
	flare_intervals: list[tuple[float, float]] | None = None,
	flare_mdp_labels: list[tuple[float, float, float]] | None = None,
	quiescent_background: float | None = None,
	flare_threshold: float | None = None,
	flux_scale: float = 1.0,
) -> None:
	"""
	This function generates a light curve plot for a given source, with options to highlight flare points and intervals, and to show quiescent background and flare threshold levels. 
	The plot is saved to the specified output path.
	"""
	time_column = _resolve_plot_column(dataframe, 'time_MJD', 'new_point_mjd')
	flux_column = _resolve_plot_column(dataframe, 'flux', 'new_point_flux')
	error_column = _resolve_plot_column(dataframe, 'flux_error', 'new_point_flux_error')
	if not np.isfinite(flux_scale) or flux_scale <= 0:
		flux_scale = 1.0
	if y_label is None:
		y_label = flux_axis_label('COSI' if INSTRUMENT == 'COSI' else 'Fermi-LAT')

	if time_range is not None:
		start_mjd, end_mjd = time_range
		dataframe = dataframe.loc[dataframe[time_column].between(start_mjd, end_mjd)].copy()
		if dataframe.empty:
			raise ValueError(f'No light-curve points found in the requested MJD range for {source_name}.')
		if flare_points is not None:
			flare_points = flare_points.loc[flare_points[time_column].between(start_mjd, end_mjd)].copy()
		if flare_intervals is not None:
			flare_intervals = [
				(max(interval_start, start_mjd), min(interval_end, end_mjd))
				for interval_start, interval_end in flare_intervals
				if interval_end >= start_mjd and interval_start <= end_mjd
			]
		if flare_mdp_labels is not None:
			flare_mdp_labels = [
				(max(interval_start, start_mjd), min(interval_end, end_mjd), mdp99)
				for interval_start, interval_end, mdp99 in flare_mdp_labels
				if interval_end >= start_mjd and interval_start <= end_mjd
			]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(PLOT_FIGURE_WIDTH, PLOT_FIGURE_HEIGHT))
	ax.errorbar(
		dataframe[time_column],
		dataframe[flux_column] * flux_scale,
		yerr=dataframe[error_column] * flux_scale,
		fmt='o-',
		color='black',
		ecolor='gray',
		linewidth=1.0,
		markersize=4,
		capsize=2,
		label='Weekly flux',
	)
	# Plotting the quiescent background if the option is toggled.
	if quiescent_background is not None and np.isfinite(quiescent_background):
		ax.axhline(
			quiescent_background * flux_scale,
			color='tab:blue',
			linestyle='--',
			linewidth=1.8,
			label='Quiescent background',
		)
	# Plotting the flare threshold if the option is toggled.
	if flare_threshold is not None and np.isfinite(flare_threshold):
		ax.axhline(
			flare_threshold * flux_scale,
			color='tab:orange',
			linestyle='--',
			linewidth=1.4,
			alpha=0.55,
			label='Flare threshold',
		)
	# If we have a flare interval, we highlight it with a shaded region. 
	if flare_intervals:
		for interval_index, (start_mjd, end_mjd) in enumerate(flare_intervals):
			ax.axvspan(
				start_mjd,
				end_mjd,
				color='orange',
				alpha=0.22,
				label='Highlighted flare interval' if interval_index == 0 else None,
			)

	if flare_mdp_labels:
		y_min, y_max = ax.get_ylim()
		y_text = y_max - (y_max - y_min) * 0.06
		for start_mjd, end_mjd, mdp99 in flare_mdp_labels:
			x_text = 0.5 * (start_mjd + end_mjd)
			ax.text(
				x_text,
				y_text,
				f'MDP99: {mdp99:.2f}%',
				fontsize=9,
				color='black',
				ha='center',
				va='bottom',
				bbox={
					'boxstyle': 'round,pad=0.2',
					'facecolor': 'white',
					'edgecolor': '#d1d5db',
					'alpha': 0.8,
				},
			)

	# If we have specific flare points, we highlight them with red markers.
	if flare_points is not None and not flare_points.empty:
		flare_time_column = _resolve_plot_column(flare_points, 'time_MJD', 'new_point_mjd')
		flare_flux_column = _resolve_plot_column(flare_points, 'flux', 'new_point_flux')
		ax.scatter(
			flare_points[flare_time_column],
			flare_points[flare_flux_column] * flux_scale,
			color='crimson',
			s=30,
			zorder=4,
			label='Flaring points',
		)

	ax.set_title(f'{source_name}: {title_suffix}')
	ax.set_xlabel('Time (MJD)')
	ax.set_ylabel(y_label)
	ax.grid(alpha=0.25)
	ax.legend(loc='upper left', frameon=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=PLOT_SAVE_DPI, bbox_inches='tight')
	plt.close(fig)




# The next few functions are just small utilities for parsing, emailing, and general data handling.
def parse_csv_list(raw_value: str, cast):
	values = []
	for item in raw_value.split(','):
		item = item.strip()
		if item:
			values.append(cast(item))
	if not values:
		raise ValueError('Expected at least one comma-separated value.')
	return values


def parse_email_recipients(raw_value: str) -> list[str]:
	return parse_csv_list(raw_value, str)


def format_optional_float(value: float | int | np.floating | None, fmt: str = '.2f', missing: str = 'n/a') -> str:
	if value is None:
		return missing
	try:
		if not np.isfinite(value):
			return missing
	except TypeError:
		return missing
	return format(float(value), fmt)


def format_scientific_optional(value: float | int | np.floating | None, missing: str = 'n/a') -> str:
	return format_optional_float(value, '.3e', missing)


def _coerce_count(value: object) -> int:
	try:
		if value is None or (isinstance(value, float) and np.isnan(value)):
			return 0
		return int(float(value))
	except (TypeError, ValueError):
		return 0


def _select_latest_mdp99(
	latest_mdp99: float | int | np.floating | None,
	latest_available: bool,
	result_df: pd.DataFrame | None,
) -> tuple[float, bool]:
	latest_value = float(latest_mdp99) if latest_mdp99 is not None and np.isfinite(latest_mdp99) else np.nan
	if np.isfinite(latest_value):
		return latest_value, bool(latest_available)
	if result_df is None or result_df.empty:
		return np.nan, False
	available_rows = result_df.loc[result_df.get('mdp99_available', False), 'mdp99_percent'].dropna()
	if available_rows.empty:
		return np.nan, False
	fallback_value = float(available_rows.iloc[-1])
	if np.isfinite(fallback_value):
		return fallback_value, True
	return np.nan, False


def _pick_first_numeric(*values: object) -> float | np.floating | None:
	for value in values:
		if value is None:
			continue
		if isinstance(value, str):
			if not value.strip():
				continue
			try:
				return float(value)
			except ValueError:
				continue
		try:
			if np.isnan(value):
				continue
		except TypeError:
			pass
		return float(value)
	return np.nan


def _should_filter_row_for_section_mdp(resolved_row: dict, mdp_column: str | None) -> bool:
	if not mdp_column:
		return False
	value = resolved_row.get(mdp_column, np.nan)
	if value is None:
		return False
	try:
		numeric_value = float(value)
	except (TypeError, ValueError):
		return False
	if not np.isfinite(numeric_value):
		return False
	return numeric_value > 100.0


def _load_saved_email_summary(row: object) -> dict | None:
	source_json_value = None
	if isinstance(row, dict):
		source_json_value = row.get('source_json') or row.get('source_json_path')
	elif hasattr(row, 'get'):
		source_json_value = row.get('source_json') or row.get('source_json_path')
	if not source_json_value:
		return None
	json_path = Path(str(source_json_value))
	if not json_path.exists():
		return None
	try:
		with json_path.open('r', encoding='utf-8') as handle:
			payload = json.load(handle)
	except (OSError, json.JSONDecodeError):
		return None
	if not isinstance(payload, dict):
		return None
	scan_state = payload.get('_scanState')
	if not isinstance(scan_state, dict):
		return None
	summary = scan_state.get('emailSummary')
	if not isinstance(summary, dict):
		return None
	return summary


def _resolve_row_with_saved_email_summary(row: object) -> dict:
	resolved = dict(row)
	saved_summary = _load_saved_email_summary(row)
	if not saved_summary:
		return resolved

	resolved['potential_points'] = _coerce_count(resolved.get('potential_points', 0)) or _coerce_count(saved_summary.get('potentialPoints', 0))
	resolved['active_flare_weeks'] = _coerce_count(resolved.get('active_flare_weeks', 0)) or _coerce_count(saved_summary.get('activeFlareWeeks', 0))
	resolved['confirmed_flare_weeks'] = _coerce_count(resolved.get('confirmed_flare_weeks', 0)) or _coerce_count(saved_summary.get('confirmedFlareWeeks', 0))
	resolved['had_potential_flare_points'] = bool(
		resolved.get('had_potential_flare_points', False)
		or _coerce_count(resolved.get('potential_points', 0)) > 0
		or _coerce_count(saved_summary.get('potentialPoints', 0)) > 0
		or bool(saved_summary.get('latestPotentialFlarePoint', False))
	)
	resolved['had_active_flare_weeks'] = bool(
		resolved.get('had_active_flare_weeks', False)
		or _coerce_count(resolved.get('active_flare_weeks', 0)) > 0
		or _coerce_count(saved_summary.get('activeFlareWeeks', 0)) > 0
		or bool(saved_summary.get('latestFlareActive', False))
	)
	resolved['had_confirmed_flare_weeks'] = bool(
		resolved.get('had_confirmed_flare_weeks', False)
		or _coerce_count(resolved.get('confirmed_flare_weeks', 0)) > 0
		or _coerce_count(saved_summary.get('confirmedFlareWeeks', 0)) > 0
		or bool(saved_summary.get('latestConfirmedFlareActive', False))
	)
	resolved['latest_potential_flare_point'] = bool(
		resolved.get('latest_potential_flare_point', False)
		or _coerce_count(resolved.get('potential_points', 0)) > 0
		or bool(saved_summary.get('latestPotentialFlarePoint', False))
		or _coerce_count(saved_summary.get('potentialPoints', 0)) > 0
	)
	resolved['latest_flare_active'] = bool(
		resolved.get('latest_flare_active', False)
		or _coerce_count(resolved.get('active_flare_weeks', 0)) > 0
		or bool(saved_summary.get('latestFlareActive', False))
		or _coerce_count(saved_summary.get('activeFlareWeeks', 0)) > 0
	)
	resolved['latest_confirmed_flare_active'] = bool(
		resolved.get('latest_confirmed_flare_active', False)
		or _coerce_count(resolved.get('confirmed_flare_weeks', 0)) > 0
		or bool(saved_summary.get('latestConfirmedFlareActive', False))
		or _coerce_count(saved_summary.get('confirmedFlareWeeks', 0)) > 0
	)
	resolved['latest_mdp99_percent'] = _pick_first_numeric(
		resolved.get('latest_mdp99_percent', np.nan),
		saved_summary.get('latestMdp99Percent', np.nan),
	)
	resolved['latest_potential_mdp99_percent'] = _pick_first_numeric(
		resolved.get('latest_potential_mdp99_percent', np.nan),
		saved_summary.get('latestPotentialMdp99Percent', np.nan),
	)
	resolved['latest_active_mdp99_percent'] = _pick_first_numeric(
		resolved.get('latest_active_mdp99_percent', np.nan),
		saved_summary.get('latestActiveMdp99Percent', np.nan),
	)
	return resolved


def _latest_flaring_downward_steps(result: pd.DataFrame) -> int:
	"""
	Count trailing consecutive downward flux steps in the latest contiguous flaring segment.
	The segment is built from rows where potential flare is true and flux is above threshold.
	"""
	if result.empty:
		return 0

	flaring = result.loc[
		result['potential_flare_point']
		& (pd.to_numeric(result['new_point_flux'], errors='coerce') > pd.to_numeric(result['flare_flux_threshold'], errors='coerce')),
		['new_point_mjd', 'new_point_flux'],
	].copy()
	if flaring.empty:
		return 0

	flaring['new_point_mjd'] = pd.to_numeric(flaring['new_point_mjd'], errors='coerce')
	flaring['new_point_flux'] = pd.to_numeric(flaring['new_point_flux'], errors='coerce')
	flaring = flaring.dropna(subset=['new_point_mjd', 'new_point_flux']).sort_values('new_point_mjd').reset_index(drop=True)
	if flaring.empty:
		return 0

	latest_full = pd.to_numeric(result['new_point_mjd'], errors='coerce')
	if latest_full.isna().all():
		return 0
	latest_mjd = float(latest_full.iloc[-1])
	if not np.isfinite(latest_mjd) or not np.isclose(float(flaring['new_point_mjd'].iloc[-1]), latest_mjd, atol=1e-6):
		return 0

	# Keep only the latest contiguous run, allowing up to 1-week gaps for missing/filtered bins.
	segment = [float(flaring['new_point_flux'].iloc[-1])]
	for idx in range(len(flaring) - 2, -1, -1):
		mjd_now = float(flaring['new_point_mjd'].iloc[idx + 1])
		mjd_prev = float(flaring['new_point_mjd'].iloc[idx])
		if (mjd_now - mjd_prev) <= 7.0 + 1e-2:
			segment.append(float(flaring['new_point_flux'].iloc[idx]))
		else:
			break
	segment = list(reversed(segment))
	if len(segment) < 2:
		return 0

	steps = 0
	for idx in range(len(segment) - 1, 0, -1):
		if segment[idx] < segment[idx - 1]:
			steps += 1
		else:
			break
	return steps


def _build_detection_rows_html(
	rows: pd.DataFrame,
	*,
	include_plot: bool,
	plot_column: str,
	active_column: str,
	mdp_column: str | None,
	inline_images: list[tuple[str, Path]],
	plot_alt_label: str,
) -> str:
	rows_html: list[str] = []
	for _, row in rows.iterrows():
		resolved_row = _resolve_row_with_saved_email_summary(row)
		name = str(resolved_row['Name'])
		is_active = bool(resolved_row[active_column])
		plot_html = '<span style="color:#6b7280;">No plot</span>'
		if include_plot:
			plot_value = str(resolved_row.get(plot_column, '') or '')
			if plot_value:
				plot_path = ROOT / plot_value
				if plot_path.exists():
					cid = make_msgid(domain='automationdetection.local')[1:-1]
					inline_images.append((cid, plot_path))
					plot_html = (
						f'<img src="cid:{cid}" alt="{html.escape(name)} {html.escape(plot_alt_label)} plot" '
						f'style="display:block;max-width:280px;width:100%;height:auto;border-radius:10px;'
						f'border:1px solid #d1d5db;" />'
					)

		if _should_filter_row_for_section_mdp(resolved_row, mdp_column):
			continue
		latest_mdp = _pick_first_numeric(
			resolved_row.get(mdp_column or 'latest_mdp99_percent', np.nan),
			resolved_row.get('latest_mdp99_percent', np.nan),
			resolved_row.get('latest_potential_mdp99_percent', np.nan),
			resolved_row.get('latest_active_mdp99_percent', np.nan),
			resolved_row.get('best_potential_mdp99_percent', np.nan),
			resolved_row.get('best_active_mdp99_percent', np.nan),
		)
		peak_value = resolved_row.get('peak_potential_flux')
		peak_flux = peak_value if np.isfinite(peak_value) else resolved_row.get('latest_new_point_flux_cosi', np.nan)
		average_value = resolved_row.get('mean_potential_flux')
		average_flux = average_value if np.isfinite(average_value) else resolved_row.get('source_average_flux', np.nan)
		threshold = resolved_row.get('latest_threshold_cosi_flux', resolved_row.get('latest_flare_flux_threshold', np.nan))

		potential_value = bool(
			resolved_row.get('had_potential_flare_points', False)
			or resolved_row.get('latest_potential_flare_point', False)
			or _coerce_count(resolved_row.get('potential_points', 0)) > 0
		)
		active_value = bool(
			resolved_row.get('had_active_flare_weeks', False)
			or resolved_row.get(active_column, False)
			or _coerce_count(resolved_row.get('active_flare_weeks', 0)) > 0
		)
		row_style = 'background:#ecfdf3;' if active_value else ''
		rows_html.append(
			f'<tr style="{row_style}">'
			f'<td>{html.escape(name)}</td>'
			f'<td>{"Yes" if potential_value else "No"}</td>'
			f'<td>{"Yes" if active_value else "No"}</td>'
			f'<td>{format_optional_float(latest_mdp)}</td>'
			f'<td>{format_scientific_optional(peak_flux)}</td>'
			f'<td>{format_scientific_optional(average_flux)}</td>'
			f'<td>{format_scientific_optional(threshold)}</td>'
			f'<td>{plot_html}</td>'
			'</tr>'
		)
	return ''.join(rows_html)




def load_factor_table(path: Path) -> pd.DataFrame:
	"""
	Load the per-source conversion-factor table used to map LAT flux into COSI count estimates.
	"""
	factors = pd.read_csv(path)
	required = {'Name', 'Aeff_mean_LAT(cm2)', 'ph/s_ratio'}
	missing = [column for column in required if column not in factors.columns]
	if missing:
		raise ValueError(f'Missing required factor-table columns in {path}: {missing}')

	factors = factors.copy()
	factors['Name'] = factors['Name'].astype(str)
	factors['Aeff_mean_LAT(cm2)'] = pd.to_numeric(factors['Aeff_mean_LAT(cm2)'], errors='coerce')
	factors['ph/s_ratio'] = pd.to_numeric(factors['ph/s_ratio'], errors='coerce')
	return factors


def _streak_start_indices(streak_counts: pd.Series) -> list[int | None]:
	starts: list[int | None] = []
	for idx, raw_streak in enumerate(streak_counts.to_list()):
		streak = int(raw_streak) if pd.notna(raw_streak) else 0
		if streak > 0:
			starts.append(idx - streak + 1)
		else:
			starts.append(None)
	return starts


def add_incremental_mdp99_columns(
	result_df: pd.DataFrame,
	factor_row: pd.Series | None,
	*,
	cosi_background_rate: float,
	arm_reduction: float,
	average_mu: float,
) -> pd.DataFrame:
	"""
	Augment incremental rows with per-row flare count estimates and MDP99.
	"""
	result = result_df.copy()
	result['flare_duration_seconds'] = np.nan
	result['streak_mean_flux'] = np.nan
	result['source_counts_0p2_5MeV'] = np.nan
	result['background_counts'] = np.nan
	result['mdp99_percent'] = np.nan
	result['mdp99_available'] = False

	if factor_row is None:
		result['mdp99_reason'] = 'missing_factor_row'
		return result

	lat_aeff = float(factor_row['Aeff_mean_LAT(cm2)'])
	ph_ratio = float(factor_row['ph/s_ratio'])
	if not np.isfinite(lat_aeff) or not np.isfinite(ph_ratio):
		result['mdp99_reason'] = 'invalid_factor_values'
		return result

	if arm_reduction <= 0 or average_mu <= 0:
		result['mdp99_reason'] = 'invalid_mdp_parameters'
		return result

	result['mdp99_reason'] = 'not_in_flare_streak'
	streak_starts = _streak_start_indices(result['consecutive_potential_flare_points'])
	new_flux = pd.to_numeric(result['new_point_flux'], errors='coerce').to_numpy(dtype=float)

	for row_index, start_index in enumerate(streak_starts):
		if start_index is None:
			continue

		segment = new_flux[start_index:row_index + 1]
		if len(segment) == 0:
			continue

		mean_flux = float(np.nanmean(segment))
		if not np.isfinite(mean_flux):
			result.at[row_index, 'mdp99_reason'] = 'non_finite_flux'
			continue

		duration_seconds = float(len(segment) * SECONDS_PER_WEEK)
		background_counts = float(duration_seconds * cosi_background_rate)
		cosi_photon_rate = float(mean_flux * lat_aeff * ph_ratio)
		source_counts = float(cosi_photon_rate * duration_seconds / arm_reduction)

		if source_counts <= 0:
			result.at[row_index, 'mdp99_reason'] = 'non_positive_source_counts'
			continue

		mdp99 = float(
			notebook_pipeline.compute_mdp99(
				source_counts,
				background_counts,
				average_mu=average_mu,
			)
		)

		result.at[row_index, 'flare_duration_seconds'] = duration_seconds
		result.at[row_index, 'streak_mean_flux'] = mean_flux
		result.at[row_index, 'source_counts_0p2_5MeV'] = source_counts
		result.at[row_index, 'background_counts'] = background_counts
		result.at[row_index, 'mdp99_percent'] = mdp99
		result.at[row_index, 'mdp99_available'] = np.isfinite(mdp99)
		result.at[row_index, 'mdp99_reason'] = 'ok' if np.isfinite(mdp99) else 'non_finite_mdp99'

	return result


def iso_week_key(today: date | None = None) -> str:
	if today is None:
		today = date.today()
	year, week, _ = today.isocalendar()
	return f'{year}-W{week:02d}'


def load_json_state(path: Path) -> dict:
	if not path.exists():
		return {}
	try:
		with path.open('r', encoding='utf-8') as handle:
			return json.load(handle)
	except Exception:
		return {}


def save_json_state(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)


def source_json_path_for_name(source_name: str) -> Path:
	safe_name = source_name.replace(' ', '_').replace('/', '_')
	return SOURCES_DIR / f'{safe_name}.json'


def cadence_window_days(cadence: str) -> float:
	key = str(cadence).strip().lower()
	if key in {'3day', '3-day', 'daily'}:
		return 3.0
	if key == 'weekly':
		return 7.0
	if key == 'monthly':
		return 30.0
	raise ValueError(f'Unsupported cadence for recency check: {cadence}')


def current_mjd_utc() -> float:
	# Unix epoch to MJD conversion: MJD = unix_seconds / 86400 + 40587.
	return datetime.now(timezone.utc).timestamp() / 86400.0 + 40587.0


def latest_source_json_mjd(path: Path) -> float:
	payload = load_json_state(path)
	if not isinstance(payload, dict):
		return np.nan
	points = payload.get('points', [])
	if not isinstance(points, list) or not points:
		return np.nan
	latest = np.nan
	for point in points:
		if not isinstance(point, dict):
			continue
		try:
			value = float(point.get('mjd', np.nan))
		except (TypeError, ValueError):
			value = np.nan
		if np.isfinite(value):
			latest = value if (not np.isfinite(latest) or value > latest) else latest
	return latest


def should_skip_source_recent_json(source_json_path: Path, cadence: str) -> tuple[bool, str]:
	if not source_json_path.exists():
		return False, 'no source JSON cache found'

	latest_mjd = latest_source_json_mjd(source_json_path)
	if not np.isfinite(latest_mjd):
		return False, 'source JSON has no valid MJD points'

	now_mjd = current_mjd_utc()
	window_days = cadence_window_days(cadence)
	age_days = float(now_mjd - latest_mjd)
	if age_days <= window_days:
		return True, f'latest JSON bin is {age_days:.2f} days old (<= {window_days:.1f} day cadence window)'
	return False, f'latest JSON bin is {age_days:.2f} days old (> {window_days:.1f} day cadence window)'


def cleanup_incremental_scan_csvs() -> tuple[int, int]:
	removed_count = 0
	removed_bytes = 0
	patterns = [
		'*_incremental_scan_thr*.csv',
		'weekly_incremental_summary_thr*.csv',
	]
	for pattern in patterns:
		for path in INCREMENTAL_OUTPUT_DIR.glob(pattern):
			if path.name == INCREMENTAL_WEEKLY_SUMMARY_PATH.name:
				continue
			if not path.is_file():
				continue
			try:
				size = path.stat().st_size
			except OSError:
				size = 0
			try:
				path.unlink()
			except OSError:
				continue
			removed_count += 1
			removed_bytes += int(size)
	return removed_count, removed_bytes


def _format_interval_label(mdp99_value: float | int | np.floating | None) -> str | None:
	if mdp99_value is None:
		return None
	try:
		if not np.isfinite(mdp99_value):
			return None
	except TypeError:
		return None
	return f'MDP99: {float(mdp99_value):.2f}%'


def _extract_mdp99_from_label(label: str | None) -> float:
	if not isinstance(label, str):
		return np.nan
	stripped = label.strip()
	if not stripped:
		return np.nan
	if stripped.lower().startswith('mdp99:'):
		stripped = stripped.split(':', 1)[1].strip()
	if stripped.endswith('%'):
		stripped = stripped[:-1].strip()
	try:
		value = float(stripped)
	except (TypeError, ValueError):
		return np.nan
	return value if np.isfinite(value) else np.nan


def _rounded_interval_key(start_mjd: float, end_mjd: float) -> tuple[float, float]:
	return (round(float(start_mjd), 6), round(float(end_mjd), 6))


def _intervals_overlap(interval_a: dict, interval_b: dict) -> bool:
	return float(interval_a['start']) <= float(interval_b['end']) and float(interval_b['start']) <= float(interval_a['end'])


def _normalize_interval_payload(interval: dict) -> dict | None:
	try:
		start_mjd = float(interval.get('start', np.nan))
		end_mjd = float(interval.get('end', np.nan))
	except (TypeError, ValueError):
		return None
	if (not np.isfinite(start_mjd)) or (not np.isfinite(end_mjd)):
		return None
	if end_mjd < start_mjd:
		start_mjd, end_mjd = end_mjd, start_mjd
	normalized = {'start': start_mjd, 'end': end_mjd}
	raw_mdp99 = interval.get('mdp99', np.nan)
	try:
		mdp99_value = float(raw_mdp99)
	except (TypeError, ValueError):
		mdp99_value = np.nan
	if not np.isfinite(mdp99_value):
		mdp99_value = np.nan
	label = interval.get('label')
	if isinstance(label, str) and label.strip():
		normalized['label'] = label.strip()
	if not np.isfinite(mdp99_value):
		mdp99_value = _extract_mdp99_from_label(normalized.get('label'))
	if np.isfinite(mdp99_value):
		normalized['mdp99'] = float(mdp99_value)
		if 'label' not in normalized:
			normalized['label'] = _format_interval_label(mdp99_value)
	return normalized


def _merge_flare_intervals(existing_intervals: list[dict], new_intervals: list[dict]) -> list[dict]:
	combined: list[dict] = []
	for interval in existing_intervals:
		normalized = _normalize_interval_payload(interval)
		if normalized is not None:
			normalized['_source'] = 'old'
			combined.append(normalized)
	for interval in new_intervals:
		normalized = _normalize_interval_payload(interval)
		if normalized is not None:
			normalized['_source'] = 'new'
			combined.append(normalized)

	if not combined:
		return []

	combined.sort(key=lambda interval: (float(interval['start']), float(interval['end']), 0 if interval.get('_source') == 'old' else 1))
	merged: list[dict] = []
	current = dict(combined[0])

	for interval in combined[1:]:
		if _intervals_overlap(current, interval):
			current['start'] = min(float(current['start']), float(interval['start']))
			current['end'] = max(float(current['end']), float(interval['end']))
			current_mdp99 = current.get('mdp99', np.nan)
			interval_mdp99 = interval.get('mdp99', np.nan)
			if interval.get('_source') == 'new' and np.isfinite(interval_mdp99):
				current['mdp99'] = float(interval_mdp99)
			elif (not np.isfinite(current_mdp99)) and np.isfinite(interval_mdp99):
				current['mdp99'] = float(interval_mdp99)
			current_label = current.get('label')
			interval_label = interval.get('label')
			if interval.get('_source') == 'new' and isinstance(interval_label, str) and interval_label.strip():
				current['label'] = interval_label.strip()
			elif (not isinstance(current_label, str) or not current_label.strip()) and isinstance(interval_label, str) and interval_label.strip():
				current['label'] = interval_label.strip()
			if np.isfinite(current.get('mdp99', np.nan)):
				current['label'] = _format_interval_label(current['mdp99'])
			if interval.get('_source') == 'new':
				current['_source'] = 'new'
			continue

		current.pop('_source', None)
		merged.append(current)
		current = dict(interval)

	current.pop('_source', None)
	merged.append(current)
	return merged


def _region_mdp99(
	points: list[dict],
	start_mjd: float,
	end_mjd: float,
	*,
	factor_row: pd.Series | None,
	cosi_background_rate: float | None,
	arm_reduction: float | None,
	average_mu: float | None,
) -> float:
	"""MDP99 over every weekly point in [start_mjd, end_mjd], using the same math as the per-row streak MDP99."""
	if factor_row is None or cosi_background_rate is None or arm_reduction is None or average_mu is None:
		return np.nan
	if arm_reduction <= 0 or average_mu <= 0:
		return np.nan
	try:
		lat_aeff = float(factor_row['Aeff_mean_LAT(cm2)'])
		ph_ratio = float(factor_row['ph/s_ratio'])
	except (KeyError, TypeError, ValueError):
		return np.nan
	if not (np.isfinite(lat_aeff) and np.isfinite(ph_ratio)):
		return np.nan

	fluxes = []
	for point in points:
		try:
			point_mjd = float(point.get('mjd', np.nan))
			point_flux = float(point.get('flux', np.nan))
		except (TypeError, ValueError):
			continue
		if np.isfinite(point_mjd) and np.isfinite(point_flux) and start_mjd - 1e-6 <= point_mjd <= end_mjd + 1e-6:
			fluxes.append(point_flux)
	if not fluxes:
		return np.nan

	duration_seconds = float(len(fluxes) * SECONDS_PER_WEEK)
	background_counts = float(duration_seconds * cosi_background_rate)
	source_counts = float(np.mean(fluxes) * lat_aeff * ph_ratio * duration_seconds / arm_reduction)
	if source_counts <= 0:
		return np.nan
	mdp99 = float(notebook_pipeline.compute_mdp99(source_counts, background_counts, average_mu=average_mu))
	return mdp99 if np.isfinite(mdp99) else np.nan


def _set_interval_mdp99(interval: dict, mdp99: float) -> None:
	if np.isfinite(mdp99):
		interval['mdp99'] = float(mdp99)
		interval['label'] = _format_interval_label(mdp99)


def _rise_intervals_from_points(points: list[dict]) -> list[dict]:
	"""Rebuild Rise-method intervals from per-point RiseMethod flags (used to backfill older JSON files)."""
	intervals: list[dict] = []
	run_start = run_end = None
	for point in sorted(points, key=lambda item: float(item.get('mjd', np.nan))):
		if bool(point.get('RiseMethod', False)):
			run_start = float(point['mjd']) if run_start is None else run_start
			run_end = float(point['mjd'])
		elif run_start is not None:
			intervals.append({'start': run_start, 'end': run_end})
			run_start = run_end = None
	if run_start is not None:
		intervals.append({'start': run_start, 'end': run_end})
	return intervals


def _union_intervals(interval_lists: list[list[dict]], point_mjds: list[float]) -> list[dict]:
	"""
	Union of interval lists. Intervals that overlap, or that have no weekly point between them,
	are one continuous flaring region.
	"""
	combined = sorted(
		(float(interval['start']), float(interval['end']))
		for interval_list in interval_lists
		for interval in interval_list
	)
	mjds = np.sort(np.asarray(point_mjds, dtype=float))
	merged: list[list[float]] = []
	for start_mjd, end_mjd in combined:
		if merged:
			last = merged[-1]
			points_between = np.count_nonzero((mjds > last[1] + 1e-6) & (mjds < start_mjd - 1e-6))
			if start_mjd <= last[1] or (mjds.size > 0 and points_between == 0):
				last[1] = max(last[1], end_mjd)
				continue
		merged.append([start_mjd, end_mjd])
	return [{'start': start_mjd, 'end': end_mjd} for start_mjd, end_mjd in merged]


def build_flare_regions(
	*,
	result: pd.DataFrame,
	factor_row: pd.Series | None,
	json_path: Path,
	flare_intervals: list[tuple[float, float]],
	flare_mdp_labels: list[tuple[float, float, float]],
	confirmed_intervals: list[tuple[float, float]] | None = None,
	cosi_background_rate: float | None = None,
	arm_reduction: float | None = None,
	average_mu: float | None = None,
) -> dict:
	"""
	Build the stored weekly points and the three interval lists (consecutive, Rise, combined) with their MDP99 values.
	The JSON writer and the email plots both use this, so they always show the same regions and labels.
	"""
	existing_payload = load_json_state(json_path)
	existing_points = existing_payload.get('points', []) if isinstance(existing_payload, dict) else []
	existing_intervals = existing_payload.get('flareIntervals', []) if isinstance(existing_payload, dict) else []

	cutoff_mjd = float(result.iloc[0]['new_point_mjd']) if not result.empty else np.nan
	retained_points = []
	for point in existing_points:
		try:
			point_mjd = float(point.get('mjd', np.nan))
		except (TypeError, ValueError):
			point_mjd = np.nan
		if np.isfinite(point_mjd) and np.isfinite(cutoff_mjd) and point_mjd < cutoff_mjd:
			retained_points.append(point)

	mdp99_lookup = {_rounded_interval_key(start_mjd, end_mjd): mdp99 for start_mjd, end_mjd, mdp99 in flare_mdp_labels}
	new_points = []
	for _, row in result.iterrows():
		rise_method = bool(row.get('confirmed_flare_active', False))
		count_method = bool(row.get('flare_active', False))
		point_is_flare = bool(row.get('potential_flare_point', False) or count_method or rise_method)
		new_points.append(
			{
				'mjd': float(row['new_point_mjd']),
				'flux': float(row['new_point_flux']),
				'error': float(row['new_point_flux_error']),
				'flare': point_is_flare,
				'RiseMethod': rise_method,
				'3CountMethod': count_method,
			}
		)

	new_intervals = []
	for start_mjd, end_mjd in flare_intervals:
		interval_mdp99 = mdp99_lookup.get(_rounded_interval_key(start_mjd, end_mjd), np.nan)
		new_interval = {
			'start': float(start_mjd),
			'end': float(end_mjd),
		}
		if np.isfinite(interval_mdp99):
			new_interval['mdp99'] = float(interval_mdp99)
			new_interval['label'] = _format_interval_label(interval_mdp99)
		new_intervals.append(new_interval)

	# Rise intervals get their own MDP99, computed over the full interval from the stored weekly points.
	new_rise_intervals = [{'start': float(start_mjd), 'end': float(end_mjd)} for start_mjd, end_mjd in confirmed_intervals or []]
	all_points = retained_points + new_points
	mdp_kwargs = {
		'factor_row': factor_row,
		'cosi_background_rate': cosi_background_rate,
		'arm_reduction': arm_reduction,
		'average_mu': average_mu,
	}

	# Files written before the per-method lists existed held consecutive-only intervals in flareIntervals.
	if isinstance(existing_payload, dict) and 'consecutiveIntervals' in existing_payload:
		existing_consecutive = existing_payload.get('consecutiveIntervals', [])
	else:
		existing_consecutive = existing_intervals
	if isinstance(existing_payload, dict) and 'riseIntervals' in existing_payload:
		existing_rise = existing_payload.get('riseIntervals', [])
	else:
		existing_rise = _rise_intervals_from_points(retained_points)

	consecutive_intervals = _merge_flare_intervals(existing_consecutive, new_intervals)
	rise_intervals = _merge_flare_intervals(existing_rise, new_rise_intervals)
	for interval in rise_intervals:
		_set_interval_mdp99(interval, _region_mdp99(all_points, float(interval['start']), float(interval['end']), **mdp_kwargs))

	# Combined regions (what the website shades): MDP99 is recomputed over the whole merged region.
	merged_intervals = _union_intervals(
		[consecutive_intervals, rise_intervals],
		[float(point['mjd']) for point in all_points if 'mjd' in point],
	)
	for interval in merged_intervals:
		start_mjd, end_mjd = float(interval['start']), float(interval['end'])
		region_mdp99 = _region_mdp99(all_points, start_mjd, end_mjd, **mdp_kwargs)
		if not np.isfinite(region_mdp99):
			inside = [
				float(item['mdp99'])
				for item in consecutive_intervals
				if start_mjd - 1e-6 <= float(item['start']) and float(item['end']) <= end_mjd + 1e-6 and np.isfinite(item.get('mdp99', np.nan))
			]
			region_mdp99 = inside[-1] if inside else np.nan
		_set_interval_mdp99(interval, region_mdp99)

	return {
		'retained_points': retained_points,
		'new_points': new_points,
		'consecutive_intervals': consecutive_intervals,
		'rise_intervals': rise_intervals,
		'merged_intervals': merged_intervals,
	}


def intervals_for_plot(
	intervals: list[dict],
	window_start: float,
	window_end: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
	"""Clip stored intervals to the plotted window so full-history regions do not stretch the x-axis."""
	spans: list[tuple[float, float]] = []
	labels: list[tuple[float, float, float]] = []
	for interval in intervals:
		start_mjd, end_mjd = float(interval['start']), float(interval['end'])
		if end_mjd < window_start or start_mjd > window_end:
			continue
		clipped = (max(start_mjd, window_start), min(end_mjd, window_end))
		spans.append(clipped)
		mdp99 = interval.get('mdp99', np.nan)
		try:
			mdp99 = float(mdp99)
		except (TypeError, ValueError):
			mdp99 = np.nan
		if np.isfinite(mdp99):
			labels.append((clipped[0], clipped[1], mdp99))
	return spans, labels


def write_source_json_output(
	*,
	source_name: str,
	result: pd.DataFrame,
	factor_row: pd.Series | None,
	json_path: Path,
	active_rows: pd.DataFrame,
	flare_intervals: list[tuple[float, float]],
	flare_mdp_labels: list[tuple[float, float, float]],
	confirmed_intervals: list[tuple[float, float]] | None = None,
	cosi_background_rate: float | None = None,
	arm_reduction: float | None = None,
	average_mu: float | None = None,
	summary_row: dict | None = None,
	regions: dict | None = None,
) -> None:
	json_path.parent.mkdir(parents=True, exist_ok=True)
	if regions is None:
		regions = build_flare_regions(
			result=result,
			factor_row=factor_row,
			json_path=json_path,
			flare_intervals=flare_intervals,
			flare_mdp_labels=flare_mdp_labels,
			confirmed_intervals=confirmed_intervals,
			cosi_background_rate=cosi_background_rate,
			arm_reduction=arm_reduction,
			average_mu=average_mu,
		)
	retained_points = regions['retained_points']
	new_points = regions['new_points']
	consecutive_intervals = regions['consecutive_intervals']
	rise_intervals = regions['rise_intervals']
	merged_intervals = regions['merged_intervals']

	latest_row = result.iloc[-1]
	background_value = float(latest_row['previous_quiescent_background']) if np.isfinite(latest_row['previous_quiescent_background']) else np.nan
	threshold_value = float(latest_row['flare_flux_threshold']) if np.isfinite(latest_row['flare_flux_threshold']) else np.nan
	if not np.isfinite(background_value):
		background_value = np.nan
	if not np.isfinite(threshold_value):
		threshold_value = np.nan

	payload = {
		'source': source_name,
		'instrument': INSTRUMENT,
		'energyRange': ENERGY_RANGE,
		'fluxUnit': 'ph cm^-2 s^-1',
		'background': background_value,
		'threshold': threshold_value,
		'points': retained_points + new_points,
		'flareIntervals': merged_intervals,
		'consecutiveIntervals': consecutive_intervals,
		'riseIntervals': rise_intervals,
		'_scanState': {
			'lastProcessedMjd': float(latest_row['new_point_mjd']),
			'lastProcessedBinCount': int(latest_row['current_bin_count']),
			'potentialStreak': int(latest_row['consecutive_potential_flare_points']),
			'potentialStreakStartMjd': float(latest_row['flare_start_mjd']) if np.isfinite(latest_row['flare_start_mjd']) else np.nan,
			'confirmedActive': bool(latest_row['confirmed_flare_active']),
			'confirmedStartMjd': float(latest_row['confirmed_flare_start_mjd']) if np.isfinite(latest_row['confirmed_flare_start_mjd']) else np.nan,
		},
	}
	if summary_row is not None:
		payload['_scanState']['emailSummary'] = {
			'potentialPoints': int(summary_row.get('potential_points', 0) or 0),
			'activeFlareWeeks': int(summary_row.get('active_flare_weeks', 0) or 0),
			'confirmedFlareWeeks': int(summary_row.get('confirmed_flare_weeks', 0) or 0),
			'latestPotentialFlarePoint': bool(summary_row.get('latest_potential_flare_point', False)),
			'latestFlareActive': bool(summary_row.get('latest_flare_active', False)),
			'latestConfirmedFlareActive': bool(summary_row.get('latest_confirmed_flare_active', False)),
			'latestMdp99Percent': float(summary_row.get('latest_mdp99_percent', np.nan)) if np.isfinite(summary_row.get('latest_mdp99_percent', np.nan)) else np.nan,
			'latestPotentialMdp99Percent': float(summary_row.get('latest_potential_mdp99_percent', np.nan)) if np.isfinite(summary_row.get('latest_potential_mdp99_percent', np.nan)) else np.nan,
			'latestActiveMdp99Percent': float(summary_row.get('latest_active_mdp99_percent', np.nan)) if np.isfinite(summary_row.get('latest_active_mdp99_percent', np.nan)) else np.nan,
		}

	if factor_row is not None:
		resolved_flux_view = resolve_flux_view(factor_row)
		if resolved_flux_view['view'] == 'COSI-scaled':
			payload['_scanState']['fluxScale'] = resolved_flux_view['flux_scale']

	with json_path.open('w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)


def sync_source_json_files_to_website() -> int:
	WEBSITE_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
	count = 0
	for json_path in sorted(SOURCES_DIR.glob('*.json')):
		shutil.copy2(json_path, WEBSITE_SOURCE_DATA_DIR / json_path.name)
		count += 1
	print(f'Copied {count} source JSON file(s) to {WEBSITE_SOURCE_DATA_DIR}')
	return count


def send_email_notification(
	*,
	smtp_host: str,
	smtp_port: int,
	use_tls: bool,
	username: str,
	password: str,
	sender: str,
	recipients: list[str],
	subject: str,
	body: str,
	html_body: str | None = None,
	inline_images: list[tuple[str, Path]] | None = None,
	attachments: list[Path] | None = None,
) -> None:
	message = EmailMessage()
	message['Subject'] = subject
	message['From'] = sender
	message['To'] = ', '.join(recipients)
	message.set_content(body)

	if html_body is not None:
		message.add_alternative(html_body, subtype='html')
		html_part = message.get_payload()[-1]
		for cid, image_path in inline_images or []:
			image_path = Path(image_path)
			if not image_path.exists():
				continue
			with image_path.open('rb') as handle:
				html_part.add_related(
					handle.read(),
					maintype='image',
					subtype='png',
					cid=cid,
					filename=image_path.name,
				)

	for attachment_path in attachments or []:
		attachment_path = Path(attachment_path)
		if not attachment_path.exists():
			continue
		with attachment_path.open('rb') as handle:
			message.add_attachment(
				handle.read(),
				maintype='image',
				subtype='png',
				filename=attachment_path.name,
			)

	try:
		with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
			if use_tls:
				server.starttls()
			if username:
				server.login(username, password)
			server.send_message(message)
	except Exception as exc:
		raise RuntimeError(f'Failed to send email via {smtp_host}:{smtp_port} to {", ".join(recipients)}: {exc}') from exc


def build_incremental_summary_for_source(
	args: argparse.Namespace,
	source_name: str,
	factor_table: pd.DataFrame,
	threshold_multiplier: float | None = None,
) -> tuple[dict, pd.DataFrame]:
	run_threshold_multiplier = float(args.flare_threshold_multiplier) if threshold_multiplier is None else float(threshold_multiplier)
	source_json_path = source_json_path_for_name(source_name)
	result = notebook_pipeline.incremental_flare_scan(
		source_name=source_name,
		database_path=Path(args.db_path),
		percent=float(args.incremental_percent),
		cadence=str(args.lightcurve_cadence),
		lookback_weeks=float(args.lookback_weeks),
		detection_method=str(args.detection_method),
		flare_threshold_multiplier=run_threshold_multiplier,
		confirmed_sigma_threshold=float(args.confirmed_sigma_threshold),
		consecutive_points=int(args.consecutive_points),
		cache_path=Path(args.qb_cache_path),
		source_json_path=source_json_path,
	)
	if result.empty:
		raise ValueError(f'No incremental rows were produced for {source_name}.')

	factor_match = factor_table.loc[factor_table['Name'] == source_name]
	factor_row = factor_match.iloc[-1] if not factor_match.empty else None
	flux_view = resolve_flux_view(factor_row)
	flux_scale = flux_view['flux_scale']
	result = add_incremental_mdp99_columns(result, factor_row, cosi_background_rate=float(args.cosi_background_rate), arm_reduction=float(args.arm_reduction), average_mu=float(args.mdp99_average_mu))

	INCREMENTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	safe_name = source_name.replace(' ', '_').replace('/', '_')
	threshold_tag = str(run_threshold_multiplier).replace('.', 'p')
	output_path = INCREMENTAL_OUTPUT_DIR / f'{safe_name}_incremental_scan_thr{threshold_tag}.csv'
	result.to_csv(output_path, index=False)

	active_rows = result.loc[result['flare_active']].copy()
	potential_rows = result.loc[result['potential_flare_point']].copy()
	confirmed_rows = result.loc[result['confirmed_flare_active']].copy()
	latest_row = result.iloc[-1]
	qb_origin = str(result['quiescent_background_origin'].iloc[0])
	cache_path_used = str(result['quiescent_background_cache_path'].iloc[0])

	latest_potential = bool(latest_row['potential_flare_point'])
	latest_active = bool(latest_row['flare_active'])
	latest_confirmed_active = bool(latest_row.get('confirmed_flare_active', False))
	latest_confirmed_sigma_delta = float(latest_row['confirmed_sigma_delta']) if np.isfinite(latest_row.get('confirmed_sigma_delta', np.nan)) else np.nan
	latest_mdp99 = float(latest_row['mdp99_percent']) if np.isfinite(latest_row.get('mdp99_percent', np.nan)) else np.nan
	latest_mdp99_available = bool(latest_row.get('mdp99_available', False))
	latest_potential_mdp99 = latest_mdp99 if latest_potential else np.nan
	flare_intervals = build_active_intervals(active_rows, start_column='flare_start_mjd')
	flare_mdp_labels = build_interval_mdp_labels(active_rows, start_column='flare_start_mjd')
	confirmed_intervals = build_active_intervals(confirmed_rows, start_column='confirmed_flare_start_mjd')

	# Same regions and MDP99 values that are written to the JSON for the website, clipped to this plot's window.
	regions = build_flare_regions(
		result=result,
		factor_row=factor_row,
		json_path=source_json_path,
		flare_intervals=flare_intervals,
		flare_mdp_labels=flare_mdp_labels,
		confirmed_intervals=confirmed_intervals,
		cosi_background_rate=float(args.cosi_background_rate),
		arm_reduction=float(args.arm_reduction),
		average_mu=float(args.mdp99_average_mu),
	)
	plot_window_start = float(result['new_point_mjd'].min())
	plot_window_end = float(result['new_point_mjd'].max())
	combined_spans, combined_labels = intervals_for_plot(regions['merged_intervals'], plot_window_start, plot_window_end)
	rise_spans, rise_labels = intervals_for_plot(regions['rise_intervals'], plot_window_start, plot_window_end)
	any_flare_points = result.loc[
		result['potential_flare_point'] | result['flare_active'] | result['confirmed_flare_active']
	].copy()

	plot_base = INCREMENTAL_OUTPUT_DIR / f'{safe_name}'
	plot_base.mkdir(parents=True, exist_ok=True)
	flare_plot_path = ''
	potential_plot_path = ''
	confirmed_plot_path = ''
	if not result.empty:
		if not potential_rows.empty or not active_rows.empty:
			flare_plot_path = str(plot_base.with_suffix('').with_name(f'{safe_name}_flare_plot.png').relative_to(ROOT))
			plot_light_curve(
				result,
				source_name,
				ROOT / flare_plot_path,
				title_suffix='incremental flare detection',
				y_label=flux_axis_label(flux_view['view']),
				flare_points=any_flare_points,
				flare_intervals=combined_spans,
				flare_mdp_labels=combined_labels,
				quiescent_background=float(latest_row['previous_quiescent_background']) if np.isfinite(latest_row.get('previous_quiescent_background', np.nan)) else None,
				flare_threshold=float(latest_row['flare_flux_threshold']) if np.isfinite(latest_row.get('flare_flux_threshold', np.nan)) else None,
				flux_scale=flux_scale,
			)
		if not potential_rows.empty:
			potential_plot_path = str(plot_base.with_suffix('').with_name(f'{safe_name}_potential_plot.png').relative_to(ROOT))
			plot_light_curve(
				result,
				source_name,
				ROOT / potential_plot_path,
				title_suffix='potential flare points',
				y_label=flux_axis_label(flux_view['view']),
				flare_points=any_flare_points,
				flare_intervals=combined_spans,
				flare_mdp_labels=combined_labels,
				quiescent_background=float(latest_row['previous_quiescent_background']) if np.isfinite(latest_row.get('previous_quiescent_background', np.nan)) else None,
				flare_threshold=float(latest_row['flare_flux_threshold']) if np.isfinite(latest_row.get('flare_flux_threshold', np.nan)) else None,
				flux_scale=flux_scale,
			)
		if not confirmed_rows.empty:
			confirmed_plot_path = str(plot_base.with_suffix('').with_name(f'{safe_name}_confirmed_plot.png').relative_to(ROOT))
			plot_light_curve(
				result,
				source_name,
				ROOT / confirmed_plot_path,
				title_suffix='confirmed flare activity',
				y_label=flux_axis_label(flux_view['view']),
				flare_points=result.loc[result['confirmed_flare_active']].copy(),
				flare_intervals=rise_spans,
				flare_mdp_labels=rise_labels,
				quiescent_background=float(latest_row['previous_quiescent_background']) if np.isfinite(latest_row.get('previous_quiescent_background', np.nan)) else None,
				flare_threshold=float(latest_row['flare_flux_threshold']) if np.isfinite(latest_row.get('flare_flux_threshold', np.nan)) else None,
				flux_scale=flux_scale,
			)
	flare_intervals = build_active_intervals(active_rows, start_column='flare_start_mjd')
	flare_mdp_labels = build_interval_mdp_labels(active_rows, start_column='flare_start_mjd')
	confirmed_intervals = build_active_intervals(confirmed_rows, start_column='confirmed_flare_start_mjd')
	print(
		f"{source_name}: weeks={len(result)}, potential_points={len(potential_rows)}, "
		f"active_flare_weeks={len(active_rows)}, latest_flux={latest_row['new_point_flux']:.3e}, "
		f"latest_threshold={latest_row['flare_flux_threshold']:.3e}, "
		f"latest_potential={latest_potential}, latest_active={latest_active}, latest_confirmed={latest_confirmed_active}, "
		f"latest_mdp99={latest_mdp99:.2f}%, qb_origin={qb_origin}"
	)
	print(f'Quiescent background cache file: {cache_path_used}')
	if factor_row is None:
		print(f'{source_name}: no factor-table row found; MDP99 is unavailable for this source.')

	latest_flare_stats = latest_highlighted_flare_stats(
		result,
		active_rows,
		factor_row,
		cosi_background_rate=float(args.cosi_background_rate),
		arm_reduction=float(args.arm_reduction),
		average_mu=float(args.mdp99_average_mu),
	)
	latest_mdp99, latest_mdp99_available = _select_latest_mdp99(
		float(latest_flare_stats['latest_highlighted_mdp99_percent']) if latest_flare_stats['latest_highlighted_mdp99_available'] else np.nan,
		bool(latest_flare_stats['latest_highlighted_mdp99_available']),
		result,
	)
	potential_mdp = potential_rows.loc[potential_rows['mdp99_available']].copy()
	best_potential_mdp99 = float(potential_mdp['mdp99_percent'].min()) if not potential_mdp.empty else np.nan
	latest_potential_mdp99 = float(potential_mdp['mdp99_percent'].iloc[-1]) if not potential_mdp.empty else latest_mdp99
	peak_potential_flux = float(latest_flare_stats['latest_highlighted_peak_flux_cosi']) if np.isfinite(latest_flare_stats['latest_highlighted_peak_flux_cosi']) else np.nan
	mean_potential_flux = float(latest_flare_stats['latest_highlighted_average_flux_cosi']) if np.isfinite(latest_flare_stats['latest_highlighted_average_flux_cosi']) else np.nan
	source_average_flux = float(latest_row['average_flux_full_series'] * flux_scale) if np.isfinite(latest_row['average_flux_full_series']) else np.nan
	latest_new_point_flux_cosi = float(latest_row['new_point_flux'] * flux_scale) if np.isfinite(latest_row['new_point_flux']) else np.nan
	latest_threshold_cosi_flux = float(latest_row['flare_flux_threshold'] * flux_scale) if np.isfinite(latest_row['flare_flux_threshold']) else np.nan
	latest_downward_steps = _latest_flaring_downward_steps(result)
	omit_from_attention = bool(latest_downward_steps >= 2)
	active_mdp = active_rows.loc[active_rows['mdp99_available']].copy()
	best_active_mdp99 = float(active_mdp['mdp99_percent'].min()) if not active_mdp.empty else np.nan
	latest_active_mdp99 = float(active_mdp.iloc[-1]['mdp99_percent']) if not active_mdp.empty else latest_mdp99
	if not active_rows.empty:
		first_active = active_rows.iloc[0]
		last_active = active_rows.iloc[-1]
		print(
			f"First active flare interval starts at MJD {first_active['flare_start_mjd']:.6f}; "
			f"latest active interval ends at MJD {last_active['new_point_mjd']:.6f}"
		)
		if np.isfinite(best_active_mdp99):
			print(
				f"{source_name}: best active-interval MDP99={best_active_mdp99:.2f}% "
				f"(latest active MDP99={latest_active_mdp99:.2f}%)"
			)

	summary_row = {
		'Name': source_name,
		'weeks': int(len(result)),
		'potential_points': int(len(potential_rows)),
		'active_flare_weeks': int(len(active_rows)),
		'had_any_detection': bool(len(potential_rows) > 0 or len(active_rows) > 0 or len(confirmed_rows) > 0),
		'had_potential_flare_points': bool(len(potential_rows) > 0),
		'had_active_flare_weeks': bool(len(active_rows) > 0),
		'had_confirmed_flare_weeks': bool(len(confirmed_rows) > 0),
		'latest_new_point_mjd': float(latest_row['new_point_mjd']),
		'latest_new_point_flux': float(latest_row['new_point_flux']),
		'latest_flare_flux_threshold': float(latest_row['flare_flux_threshold']),
		'latest_potential_flare_point': latest_potential or bool(len(potential_rows) > 0),
		'latest_flare_active': latest_active or bool(len(active_rows) > 0),
		'latest_mdp99_percent': latest_mdp99,
		'latest_mdp99_available': latest_mdp99_available,
		'latest_potential_mdp99_percent': latest_potential_mdp99,
		'best_potential_mdp99_percent': best_potential_mdp99,
		'latest_active_mdp99_percent': latest_active_mdp99,
		'best_active_mdp99_percent': best_active_mdp99,
		'peak_potential_flux': peak_potential_flux,
		'mean_potential_flux': mean_potential_flux,
		'source_average_flux': source_average_flux,
		'latest_new_point_flux_cosi': latest_new_point_flux_cosi,
		'latest_threshold_cosi_flux': latest_threshold_cosi_flux,
		'cadence_run': str(args.lightcurve_cadence),
		'lookback_weeks_run': float(args.lookback_weeks),
		'detection_method_run': str(args.detection_method),
		'confirmed_flare_weeks': int(len(confirmed_rows)),
		'latest_confirmed_flare_active': latest_confirmed_active or bool(len(confirmed_rows) > 0),
		'latest_confirmed_sigma_delta': latest_confirmed_sigma_delta,
		'flare_threshold_multiplier_run': run_threshold_multiplier,
		'latest_flaring_downward_steps': int(latest_downward_steps),
		'omit_from_attention': omit_from_attention,
		'quiescent_background_origin': qb_origin,
		'scan_csv': str(output_path.relative_to(ROOT)),
		'source_json': str(source_json_path.relative_to(ROOT)),
		'flare_plot': flare_plot_path,
		'potential_flare_plot': potential_plot_path,
		'confirmed_flare_plot': confirmed_plot_path,
	}

	write_source_json_output(
		 source_name=source_name,
		 result=result,
		 factor_row=factor_row,
		 json_path=source_json_path,
		 active_rows=active_rows,
		 flare_intervals=flare_intervals,
		 flare_mdp_labels=flare_mdp_labels,
		 confirmed_intervals=confirmed_intervals,
		 cosi_background_rate=float(args.cosi_background_rate),
		 arm_reduction=float(args.arm_reduction),
		 average_mu=float(args.mdp99_average_mu),
		 summary_row=summary_row,
		 regions=regions,
	)
	print(f'Wrote source JSON to {source_json_path.relative_to(ROOT)}')
	print(f'Wrote incremental scan to {output_path.relative_to(ROOT)}')

	return summary_row, result




def build_multi_threshold_email_content(
	detections_by_multiplier: dict[float, pd.DataFrame],
	week_key: str,
	*,
	include_potential_plots: bool,
	include_confirmed_plots: bool,
) -> tuple[str, str, list[tuple[str, Path]], int]:
	inline_images: list[tuple[str, Path]] = []
	text_lines = [f'Weekly incremental flare report ({week_key})', '']
	html_sections: list[str] = []
	total_detections = 0

	for multiplier in sorted(detections_by_multiplier.keys()):
		detections = detections_by_multiplier[multiplier].sort_values('Name').copy()
		potential_detections = []
		confirmed_detections = []
		rendered_rows_for_text = []
		for _, row in detections.sort_values('Name').iterrows():
			resolved_row = _resolve_row_with_saved_email_summary(row)
			potential_value = bool(
				resolved_row.get('had_potential_flare_points', False)
				or resolved_row.get('latest_potential_flare_point', False)
				or _coerce_count(resolved_row.get('potential_points', 0)) > 0
			)
			active_value = bool(
				resolved_row.get('had_active_flare_weeks', False)
				or resolved_row.get('latest_flare_active', False)
				or _coerce_count(resolved_row.get('active_flare_weeks', 0)) > 0
			)
			confirmed_value = bool(
				resolved_row.get('had_confirmed_flare_weeks', False)
				or resolved_row.get('latest_confirmed_flare_active', False)
				or _coerce_count(resolved_row.get('confirmed_flare_weeks', 0)) > 0
			)
			include_in_potential = (
				potential_value
				and not active_value
				and not confirmed_value
				and not _should_filter_row_for_section_mdp(resolved_row, 'latest_potential_mdp99_percent')
			)
			include_in_confirmed = (active_value or confirmed_value) and not _should_filter_row_for_section_mdp(resolved_row, 'latest_active_mdp99_percent')
			if include_in_potential:
				potential_detections.append(resolved_row)
			if include_in_confirmed:
				confirmed_detections.append(resolved_row)
			if include_in_potential or include_in_confirmed:
				rendered_rows_for_text.append(resolved_row)
		potential_detections = pd.DataFrame(potential_detections)
		confirmed_detections = pd.DataFrame(confirmed_detections)
		print(
			f'Email builder: multiplier={multiplier:g} detections={len(detections)} '
			f'potential_table_rows={len(potential_detections)} confirmed_table_rows={len(confirmed_detections)}'
		)

		text_lines.extend(
			[
				f'Threshold Multiplier = {multiplier:g}',
				f'- Processed sources: {len(detections)}',
				f'- Potential rows: {len(potential_detections)}',
				f'- Confirmed rows: {len(confirmed_detections)}',
			]
		)
		for resolved_row in rendered_rows_for_text:
			name = str(resolved_row['Name'])
			potential_value = bool(
				resolved_row.get('had_potential_flare_points', False)
				or resolved_row.get('latest_potential_flare_point', False)
				or _coerce_count(resolved_row.get('potential_points', 0)) > 0
			)
			active_value = bool(
				resolved_row.get('had_active_flare_weeks', False)
				or resolved_row.get('latest_flare_active', False)
				or _coerce_count(resolved_row.get('active_flare_weeks', 0)) > 0
			)
			confirmed_value = bool(
				resolved_row.get('had_confirmed_flare_weeks', False)
				or resolved_row.get('latest_confirmed_flare_active', False)
				or _coerce_count(resolved_row.get('confirmed_flare_weeks', 0)) > 0
			)
			latest_mdp = _pick_first_numeric(
				resolved_row.get('latest_mdp99_percent', np.nan),
				resolved_row.get('latest_potential_mdp99_percent', np.nan),
				resolved_row.get('latest_active_mdp99_percent', np.nan),
			)
			confirmed_sigma = resolved_row.get('latest_confirmed_sigma_delta', np.nan)
			peak_value = resolved_row.get('peak_potential_flux')
			peak_flux = peak_value if np.isfinite(peak_value) else resolved_row.get('latest_new_point_flux_cosi', np.nan)
			average_value = resolved_row.get('mean_potential_flux')
			average_flux = average_value if np.isfinite(average_value) else resolved_row.get('source_average_flux', np.nan)
			threshold = resolved_row.get('latest_threshold_cosi_flux', resolved_row.get('latest_flare_flux_threshold', np.nan))
			text_lines.append(
				f"  - {name}: potential={potential_value}, "
				f"active={active_value}, confirmed={confirmed_value}, "
				f"confirmed_sigma={format_optional_float(confirmed_sigma)}, "
				f"latest_mdp99={format_optional_float(latest_mdp)}, "
				f"peak_flux={format_scientific_optional(peak_flux)}, "
				f"average_flux={format_scientific_optional(average_flux)}, "
				f"threshold={format_scientific_optional(threshold)}"
			)
		text_lines.append('')
		total_detections += len(rendered_rows_for_text)

		potential_rows_html = _build_detection_rows_html(
			potential_detections,
			include_plot=include_potential_plots,
			plot_column='potential_flare_plot',
			active_column='latest_flare_active',
			mdp_column='latest_potential_mdp99_percent',
			inline_images=inline_images,
			plot_alt_label='potential flare',
		)
		confirmed_rows_html = _build_detection_rows_html(
			confirmed_detections,
			include_plot=include_confirmed_plots,
			plot_column='confirmed_flare_plot',
			active_column='latest_confirmed_flare_active',
			mdp_column='latest_active_mdp99_percent',
			inline_images=inline_images,
			plot_alt_label='confirmed flare',
		)
		potential_plot_label = 'with plots' if include_potential_plots else 'without plots'
		confirmed_plot_label = 'with plots' if include_confirmed_plots else 'without plots'

		html_sections.append(
			f'''
			<div style="background:white;border-radius:18px;padding:20px 22px;box-shadow:0 10px 30px rgba(15,23,42,0.08);overflow-x:auto;margin-bottom:18px;">
			  <div style="font-size:18px;font-weight:700;color:#111827;margin:0 0 12px 0;">Threshold Multiplier = {multiplier:g}</div>
			  <div style="font-size:16px;font-weight:700;color:#065f46;margin:0 0 10px 0;">Potential flare detections ({potential_plot_label})</div>
			  <table style="width:100%;border-collapse:collapse;font-size:14px;min-width:1100px;">
				<thead>
				  <tr style="background:#eef2ff;color:#1f2937;">
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Source</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Potential</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Active</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Latest MDP99</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Peak Flux</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Average Flux</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Threshold</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #dbe4ff;">Plot</th>
				  </tr>
				</thead>
				<tbody>{potential_rows_html}</tbody>
			  </table>
			  <div style="height:18px;"></div>
			  <div style="font-size:16px;font-weight:700;color:#1f2937;margin:0 0 10px 0;">Confirmed flare detections ({confirmed_plot_label})</div>
			  <table style="width:100%;border-collapse:collapse;font-size:14px;min-width:1100px;">
				<thead>
				  <tr style="background:#f3f4f6;color:#1f2937;">
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Source</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Potential</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Active</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Latest MDP99</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Peak Flux</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Average Flux</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Threshold</th>
					<th style="text-align:left;padding:12px 10px;border-bottom:1px solid #e5e7eb;">Plot</th>
				  </tr>
				</thead>
				<tbody>{confirmed_rows_html}</tbody>
			  </table>
			</div>
			'''
		)

	html_body = f'''
<!DOCTYPE html>
<html>
  <body style="margin:0;background:#f6f7fb;font-family:Arial,Helvetica,sans-serif;color:#111827;">
    <div style="max-width:1200px;margin:0 auto;padding:24px;">
      <div style="background:#111827;color:#f9fafb;border-radius:18px;padding:24px 28px;margin-bottom:20px;">
        <div style="font-size:13px;letter-spacing:0.08em;text-transform:uppercase;color:#93c5fd;">AutomationDetection</div>
        <div style="font-size:26px;font-weight:700;margin-top:6px;">Weekly incremental flare report</div>
        <div style="margin-top:8px;font-size:14px;color:#d1d5db;">{html.escape(week_key)} · {total_detections} detected source entries across multipliers</div>
      </div>
      {''.join(html_sections)}
    </div>
  </body>
</html>
'''

	return '\n'.join(text_lines), html_body, inline_images, total_detections


def maybe_send_weekly_detection_email_multi(
	args: argparse.Namespace,
	detections_by_multiplier: dict[float, pd.DataFrame],
	summary_paths_by_multiplier: dict[float, Path],
) -> None:
	if not args.email_on_detections:
		return
	total_detections = int(sum(len(df) for df in detections_by_multiplier.values()))
	if total_detections == 0 and not args.email_force_send:
		print('No detections this run; no email sent.')
		return

	if not args.smtp_host:
		raise ValueError('Email requested, but --smtp-host is missing.')
	if not args.email_from:
		raise ValueError('Email requested, but --email-from is missing.')
	if not args.email_to:
		raise ValueError('Email requested, but --email-to is missing.')

	recipients = parse_email_recipients(args.email_to)
	username = args.smtp_user or ''
	password = ''
	if not username:
		raise ValueError('Email requested, but SMTP username is missing.')
	password = os.environ.get(args.smtp_password_env, '')
	if not password:
		raise ValueError(
			f'Email requested and --smtp-user was provided, but env var {args.smtp_password_env} is empty.'
		)

	week_key = iso_week_key()
	print(f'Email send path: week={week_key} total_detections={total_detections} force_send={args.email_force_send}')
	text_body, html_body, inline_images, total_detections = build_multi_threshold_email_content(
		detections_by_multiplier,
		week_key,
		include_potential_plots=bool(args.email_include_potential_plots),
		include_confirmed_plots=bool(args.email_include_confirmed_plots),
	)
	multiplier_label = ', '.join(f'{x:g}' for x in sorted(detections_by_multiplier.keys()))
	subject = f"{args.email_subject_prefix} [{week_key}] {total_detections} source summaries (x{multiplier_label})"
	print(f'Email send path: subject={subject}')
	print(f'Email send path: text_body_len={len(text_body)} html_body_len={len(html_body)} inline_images={len(inline_images)}')

	send_email_notification(smtp_host=args.smtp_host, smtp_port=int(args.smtp_port), use_tls=not args.smtp_no_tls, username=username, password=password, sender=args.email_from, recipients=recipients, subject=subject, body=text_body, html_body=html_body, inline_images=inline_images)

	state_path = Path(args.email_state_path)
	state = load_json_state(state_path)
	state['last_sent_week'] = week_key
	state['last_subject'] = subject
	state['last_detection_count'] = int(total_detections)
	state['last_summary_csv'] = json.dumps({f'{k:g}': str(v) for k, v in summary_paths_by_multiplier.items()}, sort_keys=True)
	save_json_state(state_path, state)
	print(f'Sent weekly detection email for {week_key} to {len(recipients)} recipient(s).')




def run_incremental_mode(args: argparse.Namespace) -> int:
	"""
	This is the main function run by the script. 
	For each source, we run the incremental flare analysis, which checks for new weekly points.
	"""

	INCREMENTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	removed_count, removed_bytes = cleanup_incremental_scan_csvs()
	if removed_count > 0:
		print(f'Cleared {removed_count} previous incremental scan CSV file(s), freeing {removed_bytes} bytes.')

	if args.source:
		target_sources = [args.source]
	else:
		target_sources = read_source_names(PYTHON_FILES / 'NameCSV')

	if not target_sources:
		print('No sources to process in incremental mode.')
		return 1

	factor_table = load_factor_table(Path(args.factor_table))
	if str(args.detection_method).strip().lower() == 'sigma':
		threshold_multipliers = [float(args.flare_threshold_multiplier)]
		print('Detection method is sigma: skipping multiplier sweep; running a single threshold pass.')
	else:
		threshold_multipliers = parse_csv_list(args.flare_threshold_multipliers, float)

	all_summaries: list[pd.DataFrame] = []
	detections_by_multiplier: dict[float, pd.DataFrame] = {}
	summary_paths_by_multiplier: dict[float, Path] = {}
	had_source_failures = False
	total_skipped_sources = 0

	INCREMENTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	for multiplier in threshold_multipliers:
		print(f'Running incremental scan with flare threshold multiplier={multiplier:g}')
		summaries = []
		failures = []
		skipped_sources = []
		for source_name in target_sources:
			source_json_path = source_json_path_for_name(source_name)
			skip_source, skip_reason = should_skip_source_recent_json(source_json_path, str(args.lightcurve_cadence))
			if skip_source:
				print(f'{source_name}: skipped - {skip_reason}')
				skipped_sources.append(source_name)
				continue
			try:
				summary_row, _ = build_incremental_summary_for_source(
					args,
					source_name,
					factor_table,
					threshold_multiplier=float(multiplier),
				)
				summaries.append(summary_row)
			except Exception as exc:
				print(f'{source_name}: failed - {exc}')
				failures.append(source_name)
				if args.source:
					had_source_failures = True
		total_skipped_sources += len(skipped_sources)
		if not summaries:
			if skipped_sources and not failures:
				print(
					f'No sources needed processing for multiplier={multiplier:g}; '
					f'skipped {len(skipped_sources)} source(s) based on recent JSON bins.'
				)
			else:
				print(f'No sources were processed successfully for multiplier={multiplier:g}.')
			continue

		summary_df = pd.DataFrame(summaries).sort_values(
			['latest_confirmed_flare_active', 'latest_potential_flare_point', 'latest_flare_active', 'latest_new_point_flux'],
			ascending=[False, False, False, False],
		)
		all_summaries.append(summary_df)
		threshold_tag = str(float(multiplier)).replace('.', 'p')
		summary_path = INCREMENTAL_OUTPUT_DIR / f'weekly_incremental_summary_thr{threshold_tag}.csv'
		summary_df.to_csv(summary_path, index=False)
		summary_paths_by_multiplier[float(multiplier)] = summary_path
		print(f'Wrote weekly incremental summary to {summary_path.relative_to(ROOT)}')

		detections = summary_df.copy()
		if 'omit_from_attention' in detections.columns:
			omitted = detections.loc[detections['omit_from_attention']].copy()
			if not omitted.empty:
				print(
					f'Omitted from email attention (multiplier={multiplier:g}) due to >=2 latest consecutive downward flaring points above threshold: '
					+ ', '.join(omitted['Name'].astype(str).tolist())
				)
		print(
			f'Detection summary (multiplier={multiplier:g}): processed={len(summary_df)}, '
			f'email_rows={len(detections)}, failed={len(failures)}, skipped={len(skipped_sources)}'
		)
		detections_by_multiplier[float(multiplier)] = detections

	if not all_summaries:
		if total_skipped_sources > 0 and not had_source_failures:
			print('All candidate sources were skipped because their latest JSON bins are within one cadence window.')
			return 0
		print('No sources were processed successfully in incremental mode.')
		return 1

	combined_summary_df = pd.concat(all_summaries, ignore_index=True)
	combined_summary_df.to_csv(INCREMENTAL_WEEKLY_SUMMARY_PATH, index=False)
	print(f'Wrote combined weekly incremental summary to {INCREMENTAL_WEEKLY_SUMMARY_PATH.relative_to(ROOT)}')

	maybe_send_weekly_detection_email_multi(args, detections_by_multiplier, summary_paths_by_multiplier)
	#sync_source_json_files_to_website()

	if had_source_failures:
		return 1
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Run the incremental week-by-week flare-threshold scan workflow.')
	parser.add_argument('--source', help='Analyze a single source name instead of the whole NameCSV list.')
	parser.add_argument(
		'--db-path',
		default=str(PYTHON_FILES / 'new_db_Aug2025_weekly.csv'),
		help='Path to the source-of-truth light curve database.',
	)
	parser.add_argument(
		'--factor-table',
		default=str(ROOT / 'COSICSV' / 'COSI_factors_all_updated_only_logpar_flare_updated_60deg_offaxis_last.csv'),
		help='Path to the COSI conversion-factor table used by the incremental pipeline.',
	)
	parser.add_argument(
		'--incremental-percent',
		type=float,
		default=0.3,
		help='Threshold percentage used to estimate quiescent background for incremental mode.',
	)
	parser.add_argument(
		'--lightcurve-cadence',
		default=os.environ.get('LIGHTCURVE_CADENCE', 'weekly'),
		choices=['3day', 'weekly', 'monthly'],
		help='Cadence for source bins used in incremental mode: 3day, weekly, or monthly.',
	)
	parser.add_argument(
		'--lookback-weeks',
		type=float,
		default=float(os.environ.get('LOOKBACK_WEEKS', '10.0')),
		help='Retained as scan-output metadata; quiescent backgrounds use the full filtered light curve.',
	)
	parser.add_argument(
		'--detection-method',
		default=os.environ.get('DETECTION_METHOD', 'both'),
		choices=['original', 'sigma', 'both'],
		help='Choose detection logic: original threshold streaks, sigma-confirmed flares, or both.',
	)
	parser.add_argument(
		'--flare-threshold-multiplier',
		type=float,
		default=2.0,
		help='A new weekly point is marked as a potential flare point when it exceeds this multiple of the previous quiescent background.',
	)
	parser.add_argument(
		'--flare-threshold-multipliers',
		default=os.environ.get('FLARE_THRESHOLD_MULTIPLIERS', '1,2,3'),
		help='Comma-separated flare-threshold multipliers to run in incremental mode (results are combined into one email).',
	)
	parser.add_argument(
		'--consecutive-points',
		type=int,
		default=3,
		help='Number of consecutive potential flare points needed before the interval is marked as a flare.',
	)
	parser.add_argument(
		'--confirmed-sigma-threshold',
		type=float,
		default=float(os.environ.get('CONFIRMED_SIGMA_THRESHOLD', '2.0')),
		help='Minimum sigma significance above quiescent background needed to start a confirmed flare.',
	)
	parser.add_argument(
		'--qb-cache-path',
		default=str(ROOT / 'DownloadedLC' / 'incremental' / 'quiescent_background_cache.csv'),
		help='CSV file used to store and reuse per-source quiescent background values for incremental mode.',
	)
	parser.add_argument(
		'--cosi-background-rate',
		type=float,
		default=float(os.environ.get('COSI_BACKGROUND_RATE', str(COSI_BACKGROUND_RATE))),
		help='COSI background rate in counts/s used to estimate background counts in MDP99.',
	)
	parser.add_argument(
		'--arm-reduction',
		type=float,
		default=float(os.environ.get('ARM_REDUCTION', str(ARM_reduction))),
		help='Source-count reduction factor used in MDP99 count estimates (default follows COSI background setting).',
	)
	parser.add_argument(
		'--mdp99-average-mu',
		type=float,
		default=float(os.environ.get('MDP99_AVERAGE_MU', str(MDP99_AVERAGE_MU))),
		help='Average modulation factor mu used in MDP99 = (4.29/(mu*Nsrc))*sqrt(Nsrc+Nbkg)*100.',
	)
	parser.add_argument(
		'--email-on-detections',
		action='store_true',
		help='Send one summary email if any sources are detected in incremental mode.',
	)
	parser.add_argument(
		'--smtp-host',
		default=os.environ.get('SMTP_HOST', ''),
		help='SMTP host used for detection emails.',
	)
	parser.add_argument(
		'--smtp-port',
		type=int,
		default=int(os.environ.get('SMTP_PORT', '587')),
		help='SMTP port used for detection emails.',
	)
	parser.add_argument(
		'--smtp-user',
		default=os.environ.get('SMTP_USER', ''),
		help='SMTP username. Leave empty if your SMTP relay does not require authentication.',
	)
	parser.add_argument(
		'--smtp-password-env',
		default='SMTP_PASSWORD',
		help='Environment variable name holding the SMTP password.',
	)
	parser.add_argument(
		'--smtp-no-tls',
		action='store_true',
		help='Disable STARTTLS when connecting to SMTP.',
	)
	parser.add_argument(
		'--email-from',
		default=os.environ.get('ALERT_EMAIL_FROM', ''),
		help='Sender email address for detection notifications.',
	)
	parser.add_argument(
		'--email-to',
		default=os.environ.get('ALERT_EMAIL_TO', ''),
		help='Comma-separated recipients for detection notifications.',
	)
	parser.add_argument(
		'--email-subject-prefix',
		default='AutomationDetection weekly flare alert',
		help='Email subject prefix for detection notifications.',
	)
	parser.add_argument(
		'--email-include-potential-plots',
		action=argparse.BooleanOptionalAction,
		default=os.environ.get('EMAIL_INCLUDE_POTENTIAL_PLOTS', '1').strip().lower() not in {'0', 'false', 'no'},
		help='Include potential-flare plots inline in the weekly email (use --no-email-include-potential-plots to disable).',
	)
	parser.add_argument(
		'--email-include-confirmed-plots',
		action=argparse.BooleanOptionalAction,
		default=os.environ.get('EMAIL_INCLUDE_CONFIRMED_PLOTS', '1').strip().lower() not in {'0', 'false', 'no'},
		help='Include confirmed-flare plots inline in the weekly email (use --no-email-include-confirmed-plots to disable).',
	)
	parser.add_argument(
		'--email-state-path',
		default=str(ROOT / 'DownloadedLC' / 'incremental' / 'weekly_email_state.json'),
		help='State file used to enforce one email per ISO week.',
	)
	parser.add_argument(
		'--email-force-send',
		action='store_true',
		help='Send email even if one has already been sent this ISO week.',
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	return run_incremental_mode(args)


if __name__ == '__main__':
	raise SystemExit(main())