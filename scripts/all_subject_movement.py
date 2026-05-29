# %% [markdown]
#
# # Quantifying movement in a single subject
#
# Here we take a look at movement data from a single subject before and
# after reversal.

# %%
import json
from pathlib import Path
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import hvplot.pandas

hv.extension("bokeh")
import panel

panel.extension(comms="vscode")
import cv2

# %% [markdown]
#
# Start by defining some variables about the data we're working with.

# %%
subject = "bianca"
session = "RR20rev.01"
task = "RR20rev"
acq = "A"
rawdata_path = Path("../rawdata")


# %% [markdown]
#
# All our files are in a common format with just a different suffix for
# the content of the file. e.g.
# `rawdata/sub-bianca/ses-RR20rev.01/sub-bianca_ses-RR20rev.01_task-RR20rev_acq-A`
# is common and then we have different suffixes for the video, the DLC
# tracking, the events, etc.
#
# Let's define a function to generate file paths.


# %%
def get_file_path(subject, session, task, acq, suffix):
    return (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-{acq}{suffix}"
    )


# %%
def load_track_session(subject, session, task, acq):
    """Load one session analysed by DeepLabCut into a DataFrame"""
    data_path = get_file_path(
        subject,
        session,
        task,
        acq,
        "DLC_HrnetW32_medass_topviewmouseAug7shuffle3_detector_best-170_snapshot_best-170.h5",
    )
    frame_times_path = get_file_path(subject, session, task, acq, "_sync.csv")
    json_path = get_file_path(subject, session, task, acq, ".json")
    with open(json_path) as f:
        recording_info = pd.Series(json.load(f))
    try:
        df = pd.read_hdf(data_path)
        frame_times = pd.read_csv(frame_times_path, header=None, names=["time"])
        df.index = pd.to_timedelta(frame_times["time"], unit="s")
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return None
    df.columns = df.columns.droplevel(0)
    df.index = df.index - df.index[recording_info.loc["leverin"] - 1]
    df = df.loc[pd.Timedelta(0, unit="s") :]
    df = df.stack(0)
    df = (
        df.mask(df["likelihood"] < 0.6)
        .unstack()
        .rolling(window=pd.Timedelta(seconds=0.2), min_periods=1)
        .median()
        .drop(columns="likelihood")
    )
    return df


def load_events_session(subject, session, task, acq):
    """Load one session analysed by DeepLabCut into a DataFrame"""
    json_path = get_file_path(subject, session, task, acq, ".json")
    with open(json_path) as f:
        recording_info = pd.Series(json.load(f))
    events_path = get_file_path(subject, session, task, acq, "_events.csv")
    try:
        df = pd.read_csv(events_path)
    except FileNotFoundError:
        print(f"File not found: {events_path}")
        return None
    df = df.replace({"lp": "llp" if recording_info["block"][0] == "L" else "rlp"})
    df["onset"] = pd.to_timedelta(df["onset"], unit="s")
    return df.fillna(0.01).set_index("onset")


def get_event_windows(df, onset, window_range=(-0.5, 0.0)):
    nearest_i = np.argmin(np.abs(df.index.get_level_values("time") - onset))
    # nearest_frame = df.index.get_level_values("frame_id")[nearest_i]
    frame_range = np.rint(np.array(window_range) * 30).astype(int)
    fs, fe = frame_range + nearest_i
    ev_window = df.iloc[fs:fe, :].copy()
    ev_window["window_offset"] = np.arange(
        frame_range[0] + 1, frame_range[1] + 1, dtype=int
    )
    return ev_window.set_index("window_offset")


# %%
subjects = ["bianca"]
sessions = [
    ("RR20prerev.02", "RR20prerev"),
    ("RR20prerev.03", "RR20prerev"),
    ("RR20rev.01", "RR20rev"),
]

# %%
tracking_dict = {
    (subject, session, acq): load_track_session(subject, session, task, acq)
    for subject in subjects
    for session, task in sessions
    for acq in ["A", "B"]
}
tracking = pd.concat(tracking_dict, names=["subject", "session", "acq"])
tracking

# %%
events_dict = {
    (subject, session, acq): load_events_session(subject, session, task, acq)
    for subject in subjects
    for session, task in sessions
    for acq in ["A", "B"]
}
events = pd.concat(events_dict, names=["subject", "session", "acq"])
events

# %%
# To get one subject
events.loc[("bianca")]

# %%
# Or one subject and one session
events.loc[("bianca", "RR20rev.01")]

# %%
event_cols = ["subject", "session", "acq", "onset", "event_id"]
lp_events_df = events.query('event_id.isin(["llp", "rlp"])').reset_index()
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(tracking.loc[(x.iloc[0].subject, x.iloc[0].session, x.iloc[0].acq)], x.iloc[0].onset)
)
lp_beta = lp_windows_df.unstack("window_offset").dropna()
lp_beta
