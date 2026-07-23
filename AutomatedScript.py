import argparse
import importlib
import html
import json
import os
import smtplib
import sys
from datetime import date
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
INCREMENTAL_WEEKLY_SUMMARY_PATH = INCREMENTAL_OUTPUT_DIR / 'weekly_incremental_summary.csv'

sys.path.insert(0, str(PYTHON_FILES))

notebook_pipeline = importlib.import_module('notebook_pipeline')

# We check to see what the defined background rate is in the env file, and use the appropriate scaling factor
# to ensure consistent background assumptions in incremental processing.
COSI_BACKGROUND_RATE = float(os.environ.get('COSI_BACKGROUND_RATE', '0.25'))

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

	flux_scale = float(factor_row['Int_flux_ratio']) if factor_row is not None and 'Int_flux_ratio' in factor_row.index else 1.0
	if not np.isfinite(flux_scale) or flux_scale <= 0:
		flux_scale = 1.0

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


def plot_light_curve(
	dataframe: pd.DataFrame,
	source_name: str,
	output_path: Path,
	title_suffix: str,
	y_label: str = r'Photon Flux (ph cm$^{-2}$ s$^{-1}$)',
	time_range: tuple[float, float] | None = None,
	flare_points: pd.DataFrame | None = None,
	flare_intervals: list[tuple[float, float]] | None = None,
	flare_mdp_labels: list[tuple[float, float, float]] | None = None,
	quiescent_background: float | None = None,
	flare_threshold: float | None = None,
) -> None:
	"""
	This function generates a light curve plot for a given source, with options to highlight flare points and intervals, and to show quiescent background and flare threshold levels. 
	The plot is saved to the specified output path.
	"""
	if time_range is not None:
		start_mjd, end_mjd = time_range
		dataframe = dataframe.loc[dataframe['time_MJD'].between(start_mjd, end_mjd)].copy()
		if dataframe.empty:
			raise ValueError(f'No light-curve points found in the requested MJD range for {source_name}.')
		if flare_points is not None:
			flare_points = flare_points.loc[flare_points['time_MJD'].between(start_mjd, end_mjd)].copy()
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
		dataframe['time_MJD'],
		dataframe['flux'],
		yerr=dataframe['flux_error'],
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
			quiescent_background,
			color='tab:blue',
			linestyle='--',
			linewidth=1.8,
			label='Quiescent background',
		)
	# Plotting the flare threshold if the option is toggled.
	if flare_threshold is not None and np.isfinite(flare_threshold):
		ax.axhline(
			flare_threshold,
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
		ax.scatter(
			flare_points['time_MJD'],
			flare_points['flux'],
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
	inline_images: list[tuple[str, Path]],
	plot_alt_label: str,
) -> str:
	rows_html: list[str] = []
	for _, row in rows.iterrows():
		name = str(row['Name'])
		is_active = bool(row[active_column])
		plot_html = '<span style="color:#6b7280;">No plot</span>'
		if include_plot:
			plot_value = str(row.get(plot_column, '') or '')
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

		latest_mdp = row.get('latest_mdp99_percent', np.nan)
		peak_value = row.get('peak_potential_flux')
		peak_flux = peak_value if np.isfinite(peak_value) else row.get('latest_new_point_flux_cosi', np.nan)
		average_value = row.get('mean_potential_flux')
		average_flux = average_value if np.isfinite(average_value) else row.get('source_average_flux', np.nan)
		threshold = row.get('latest_threshold_cosi_flux', row.get('latest_flare_flux_threshold', np.nan))

		row_style = 'background:#ecfdf3;' if is_active else ''
		rows_html.append(
			f'<tr style="{row_style}">'
			f'<td>{html.escape(name)}</td>'
			f'<td>{"Yes" if bool(row["latest_potential_flare_point"]) else "No"}</td>'
			f'<td>{"Yes" if is_active else "No"}</td>'
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

	with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
		if use_tls:
			server.starttls()
		if username:
			server.login(username, password)
		server.send_message(message)


def build_incremental_summary_for_source(
	args: argparse.Namespace,
	source_name: str,
	factor_table: pd.DataFrame,
	threshold_multiplier: float | None = None,
) -> tuple[dict, pd.DataFrame]:
	run_threshold_multiplier = float(args.flare_threshold_multiplier) if threshold_multiplier is None else float(threshold_multiplier)
	result = notebook_pipeline.incremental_flare_scan(source_name=source_name, database_path=Path(args.db_path), percent=float(args.incremental_percent), cadence=str(args.lightcurve_cadence), lookback_weeks=float(args.lookback_weeks), detection_method=str(args.detection_method), flare_threshold_multiplier=run_threshold_multiplier, confirmed_sigma_threshold=float(args.confirmed_sigma_threshold), consecutive_points=int(args.consecutive_points), cache_path=Path(args.qb_cache_path))
	if result.empty:
		raise ValueError(f'No incremental rows were produced for {source_name}.')

	factor_match = factor_table.loc[factor_table['Name'] == source_name]
	factor_row = factor_match.iloc[-1] if not factor_match.empty else None
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
	latest_mdp99 = float(latest_row['mdp99_percent']) if np.isfinite(latest_row['mdp99_percent']) else np.nan
	latest_mdp99_available = bool(latest_row['mdp99_available'])
	latest_potential_mdp99 = latest_mdp99 if latest_potential else np.nan
	potential_plot_path = ''
	confirmed_plot_path = ''
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

	flux_scale = float(factor_row['Int_flux_ratio']) if factor_row is not None and 'Int_flux_ratio' in factor_row.index else 1.0
	if not np.isfinite(flux_scale) or flux_scale <= 0:
		flux_scale = 1.0
	latest_flare_stats = latest_highlighted_flare_stats(
		result,
		active_rows,
		factor_row,
		cosi_background_rate=float(args.cosi_background_rate),
		arm_reduction=float(args.arm_reduction),
		average_mu=float(args.mdp99_average_mu),
	)
	latest_mdp99 = float(latest_flare_stats['latest_highlighted_mdp99_percent']) if latest_flare_stats['latest_highlighted_mdp99_available'] else np.nan
	latest_mdp99_available = bool(latest_flare_stats['latest_highlighted_mdp99_available'])
	potential_mdp = potential_rows.loc[potential_rows['mdp99_available']].copy()
	best_potential_mdp99 = float(potential_mdp['mdp99_percent'].min()) if not potential_mdp.empty else np.nan
	latest_potential_mdp99 = latest_mdp99
	peak_potential_flux = float(latest_flare_stats['latest_highlighted_peak_flux_cosi']) if np.isfinite(latest_flare_stats['latest_highlighted_peak_flux_cosi']) else np.nan
	mean_potential_flux = float(latest_flare_stats['latest_highlighted_average_flux_cosi']) if np.isfinite(latest_flare_stats['latest_highlighted_average_flux_cosi']) else np.nan
	source_average_flux = float(latest_row['average_flux_full_series'] * flux_scale) if np.isfinite(latest_row['average_flux_full_series']) else np.nan
	latest_new_point_flux_cosi = float(latest_row['new_point_flux'] * flux_scale) if np.isfinite(latest_row['new_point_flux']) else np.nan
	latest_threshold_cosi_flux = float(latest_row['flare_flux_threshold'] * flux_scale) if np.isfinite(latest_row['flare_flux_threshold']) else np.nan
	latest_downward_steps = _latest_flaring_downward_steps(result)
	omit_from_attention = bool(latest_downward_steps >= 2)
	active_mdp = active_rows.loc[active_rows['mdp99_available']].copy()
	best_active_mdp99 = float(active_mdp['mdp99_percent'].min()) if not active_mdp.empty else np.nan
	latest_active_mdp99 = float(active_mdp.iloc[-1]['mdp99_percent']) if not active_mdp.empty else np.nan
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

	if latest_active or latest_potential or latest_confirmed_active:
		cadence_df = notebook_pipeline.load_weekly_database(Path(args.db_path), cadence=str(args.lightcurve_cadence))
		_, time_values, flux_values, error_values, _ = notebook_pipeline.build_source_arrays(cadence_df, source_name)
		plot_scale_factor = 1.0
		y_axis_label = r'Photon Flux (ph cm$^{-2}$ s$^{-1}$)'
		if factor_row is not None:
			int_flux_ratio = float(factor_row['Int_flux_ratio']) if 'Int_flux_ratio' in factor_row.index else np.nan
			if np.isfinite(int_flux_ratio) and int_flux_ratio > 0:
				plot_scale_factor = float(int_flux_ratio)
				y_axis_label = r'COSI-band Photon Flux (ph cm$^{-2}$ s$^{-1}$)'
		plot_df = pd.DataFrame({'time_MJD': time_values, 'flux': flux_values * plot_scale_factor, 'flux_error': error_values * plot_scale_factor})
		lookback_days = float(args.lookback_weeks) * 7.0
		window_end_mjd = float(plot_df['time_MJD'].max())
		window_start_target = window_end_mjd - lookback_days
		window_df = plot_df.loc[plot_df['time_MJD'] >= window_start_target].copy()
		if window_df.empty:
			window_df = plot_df.tail(1).copy()
		window_size = int(len(window_df))
		window_start_mjd = float(window_df['time_MJD'].iloc[0])
		window_end_mjd = float(window_df['time_MJD'].iloc[-1])
		potential_mjds = result.loc[
			result['potential_flare_point'] | result['flare_active'],
			'new_point_mjd',
		].astype(float)
		potential_points_plot = plot_df.loc[plot_df['time_MJD'].isin(set(potential_mjds.tolist()))].copy()
		if latest_active or latest_potential:
			potential_plot_path_obj = INCREMENTAL_OUTPUT_DIR / f'{safe_name}_potential_last_{int(float(args.lookback_weeks))}w_thr{threshold_tag}.png'
			plot_light_curve(
				plot_df, source_name, potential_plot_path_obj, f'Potential flare view (last {float(args.lookback_weeks):g} weeks, {window_size} bins, COSI flux-scaled)',
				y_label=y_axis_label, time_range=(window_start_mjd, window_end_mjd), flare_points=potential_points_plot,
				flare_intervals=flare_intervals, flare_mdp_labels=flare_mdp_labels,
				quiescent_background=float(latest_row['previous_quiescent_background']) * plot_scale_factor if np.isfinite(latest_row['previous_quiescent_background']) else None,
				flare_threshold=float(latest_row['flare_flux_threshold']) * plot_scale_factor if np.isfinite(latest_row['flare_flux_threshold']) else None,
			)
			potential_plot_path = str(potential_plot_path_obj.relative_to(ROOT))
			print(f'Wrote potential flare plot to {potential_plot_path}')

		confirmed_mjds = result.loc[
			result['confirmed_flare_active'] | result['confirmed_start_trigger'],
			'new_point_mjd',
		].astype(float)
		confirmed_points_plot = plot_df.loc[plot_df['time_MJD'].isin(set(confirmed_mjds.tolist()))].copy()
		if latest_confirmed_active:
			confirmed_plot_path_obj = INCREMENTAL_OUTPUT_DIR / f'{safe_name}_confirmed_last_{int(float(args.lookback_weeks))}w_thr{threshold_tag}.png'
			plot_light_curve(
				plot_df, source_name, confirmed_plot_path_obj, f'Confirmed flare view (last {float(args.lookback_weeks):g} weeks, {window_size} bins, COSI flux-scaled)',
				y_label=y_axis_label, time_range=(window_start_mjd, window_end_mjd), flare_points=confirmed_points_plot,
				flare_intervals=confirmed_intervals, flare_mdp_labels=None,
				quiescent_background=float(latest_row['previous_quiescent_background']) * plot_scale_factor if np.isfinite(latest_row['previous_quiescent_background']) else None,
				flare_threshold=None,
			)
			confirmed_plot_path = str(confirmed_plot_path_obj.relative_to(ROOT))
			print(f'Wrote confirmed flare plot to {confirmed_plot_path}')
	else:
		print('No potential, active, or confirmed flare points were available for plotting.')
	print(f'Wrote incremental scan to {output_path.relative_to(ROOT)}')

	summary_row = {
		'Name': source_name,
		'weeks': int(len(result)),
		'potential_points': int(len(potential_rows)),
		'active_flare_weeks': int(len(active_rows)),
		'latest_new_point_mjd': float(latest_row['new_point_mjd']),
		'latest_new_point_flux': float(latest_row['new_point_flux']),
		'latest_flare_flux_threshold': float(latest_row['flare_flux_threshold']),
		'latest_potential_flare_point': latest_potential,
		'latest_flare_active': latest_active,
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
		'latest_confirmed_flare_active': latest_confirmed_active,
		'latest_confirmed_sigma_delta': latest_confirmed_sigma_delta,
		'flare_threshold_multiplier_run': run_threshold_multiplier,
		'latest_flaring_downward_steps': int(latest_downward_steps),
		'omit_from_attention': omit_from_attention,
		'quiescent_background_origin': qb_origin,
		'scan_csv': str(output_path.relative_to(ROOT)),
		'flare_plot': potential_plot_path,
		'potential_flare_plot': potential_plot_path,
		'confirmed_flare_plot': confirmed_plot_path,
	}
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
		total_detections += len(detections)
		potential_detections = detections.loc[detections['latest_potential_flare_point']].sort_values('Name').copy()
		confirmed_detections = detections.loc[detections['latest_confirmed_flare_active']].sort_values('Name').copy()

		text_lines.extend(
			[
				f'Threshold Multiplier = {multiplier:g}',
				f'- Detected sources: {len(detections)}',
				f'- Potential detections: {len(potential_detections)}',
				f'- Confirmed detections: {len(confirmed_detections)}',
			]
		)
		for _, row in detections.iterrows():
			name = str(row['Name'])
			latest_mdp = row.get('latest_mdp99_percent', np.nan)
			confirmed_sigma = row.get('latest_confirmed_sigma_delta', np.nan)
			peak_value = row.get('peak_potential_flux')
			peak_flux = peak_value if np.isfinite(peak_value) else row.get('latest_new_point_flux_cosi', np.nan)
			average_value = row.get('mean_potential_flux')
			average_flux = average_value if np.isfinite(average_value) else row.get('source_average_flux', np.nan)
			threshold = row.get('latest_threshold_cosi_flux', row.get('latest_flare_flux_threshold', np.nan))
			text_lines.append(
				f"  - {name}: potential={bool(row['latest_potential_flare_point'])}, "
				f"active={bool(row['latest_flare_active'])}, confirmed={bool(row['latest_confirmed_flare_active'])}, "
				f"confirmed_sigma={format_optional_float(confirmed_sigma)}, "
				f"latest_mdp99={format_optional_float(latest_mdp)}, "
				f"peak_flux={format_scientific_optional(peak_flux)}, "
				f"average_flux={format_scientific_optional(average_flux)}, "
				f"threshold={format_scientific_optional(threshold)}"
			)
		text_lines.append('')

		potential_rows_html = _build_detection_rows_html(
			potential_detections,
			include_plot=include_potential_plots,
			plot_column='potential_flare_plot',
			active_column='latest_flare_active',
			inline_images=inline_images,
			plot_alt_label='potential flare',
		)
		confirmed_rows_html = _build_detection_rows_html(
			confirmed_detections,
			include_plot=include_confirmed_plots,
			plot_column='confirmed_flare_plot',
			active_column='latest_confirmed_flare_active',
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
	if total_detections == 0:
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
	if username:
		password = os.environ.get(args.smtp_password_env, '')
		if not password:
			raise ValueError(
				f'Email requested and --smtp-user was provided, but env var {args.smtp_password_env} is empty.'
			)

	state_path = Path(args.email_state_path)
	state = load_json_state(state_path)
	week_key = iso_week_key()
	already_sent = state.get('last_sent_week') == week_key

	if already_sent and not args.email_force_send:
		print(f'Email already sent for {week_key}; skipping this run.')
		return

	text_body, html_body, inline_images, total_detections = build_multi_threshold_email_content(
		detections_by_multiplier,
		week_key,
		include_potential_plots=bool(args.email_include_potential_plots),
		include_confirmed_plots=bool(args.email_include_confirmed_plots),
	)
	multiplier_label = ', '.join(f'{x:g}' for x in sorted(detections_by_multiplier.keys()))
	subject = f"{args.email_subject_prefix} [{week_key}] {total_detections} source entries detected (x{multiplier_label})"

	send_email_notification(smtp_host=args.smtp_host, smtp_port=int(args.smtp_port), use_tls=not args.smtp_no_tls, username=username, password=password, sender=args.email_from, recipients=recipients, subject=subject, body=text_body, html_body=html_body, inline_images=inline_images)

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

	INCREMENTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	for multiplier in threshold_multipliers:
		print(f'Running incremental scan with flare threshold multiplier={multiplier:g}')
		summaries = []
		failures = []
		for source_name in target_sources:
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
		if not summaries:
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

		detections = summary_df.loc[
			summary_df['latest_potential_flare_point']
			| summary_df['latest_flare_active']
			| summary_df['latest_confirmed_flare_active']
		].copy()
		if 'omit_from_attention' in detections.columns:
			omitted = detections.loc[detections['omit_from_attention']].copy()
			detections = detections.loc[~detections['omit_from_attention']].copy()
			if not omitted.empty:
				print(
					f'Omitted from email attention (multiplier={multiplier:g}) due to >=2 latest consecutive downward flaring points above threshold: '
					+ ', '.join(omitted['Name'].astype(str).tolist())
				)
		print(
			f'Detection summary (multiplier={multiplier:g}): processed={len(summary_df)}, '
			f'detected={len(detections)}, failed={len(failures)}'
		)
		detections_by_multiplier[float(multiplier)] = detections

	if not all_summaries:
		print('No sources were processed successfully in incremental mode.')
		return 1

	combined_summary_df = pd.concat(all_summaries, ignore_index=True)
	combined_summary_df.to_csv(INCREMENTAL_WEEKLY_SUMMARY_PATH, index=False)
	print(f'Wrote combined weekly incremental summary to {INCREMENTAL_WEEKLY_SUMMARY_PATH.relative_to(ROOT)}')

	maybe_send_weekly_detection_email_multi(args, detections_by_multiplier, summary_paths_by_multiplier)

	if had_source_failures:
		return 1
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Run the incremental week-by-week flare-threshold scan workflow.')
	parser.add_argument('--source', help='Analyze a single source name instead of the whole NameCSV list.')
	parser.add_argument(
		'--db-path',
		default=str(PYTHON_FILES / 'new_db_Aug2025.csv'),
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
		default=float(os.environ.get('LOOKBACK_WEEKS', '30.0')),
		help='Only analyze bins within this many weeks from the latest cadence point.',
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