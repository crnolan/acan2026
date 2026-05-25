# %% [markdown]
#
# # Quantifying movement in a single subject
#
# Here we take a look at movement data from a single subject before and
# after reversal.

# %% [markdown]
#
#

# %%
import json
from pathlib import Path
import pandas as pd
import numpy as np
import hvplot.pandas

# %% [markdown]
#
# Start by defining some variables about the data we're working with.

# %%
subject = "bianca"
session = "RR20rev.01"
task = "RR20"
rawdata_path = Path("../rawdata")

# %% [markdown]
#
# Now we can load up some metadata about our acquisitions.

# %%

json_paths = {
    "A": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-A.json"
    ),
    "B": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-B.json"
    ),
}
recordings_dict = {}
for acq, json_path in json_paths.items():
    with open(json_path) as f:
        recordings_dict[acq] = pd.Series(json.load(f))
recordings = pd.DataFrame(recordings_dict).T
recordings.index.name = "acq"
recordings

# %% [markdown]
#
# firston and laston are the times of the first and last frames


# %% [markdown]
#
# We can read in a DeepLabCut tracking file (HDF5 format) using the
# pandas read_hdf function.

# %%

dlc_path = (
    rawdata_path
    / f"sub-{subject}"
    / f"ses-{session}"
    / f"sub-{subject}_ses-{session}_task-{task}_acq-ADLC_HrnetW32_medass_topviewmouseAug7shuffle3_detector_best-170_snapshot_best-170.h5"
)
tracking = pd.read_hdf(dlc_path)
tracking

# %% [markdown]
#
# It initially looks like there is no data, but that is the very start
# and end of the tracking session. Let's look in the middle.
#
# `len(tracking)` gives the number of rows in the table, and the `iloc`
# method allows us to access rows by index. We can use the notation
# `start:end` to request a range of rows.

# %%
# This will give us the middle(ish) 10 rows
tracking.iloc[(len(tracking) // 2) : ((len(tracking) // 2) + 10)]

# %% [markdown]
#
# There is a strange "scorer" level in the columns, we don't need that
# so let's drop it.

# %%
tracking.columns = tracking.columns.droplevel(0)
tracking

# %% [markdown]
#
# The index currently doesn't have an identifier, let's call it "frame_id".

# %%
tracking.index.name = "frame_id"
tracking


# %% [markdown]
#
# This data has already been filtered based on the likelihood column
# using a median filter. We can just drop the likelihood column. At the
# same time, we can filter out all the frames outside the first and last
# LED flashes.

# %%
tracking = tracking.loc[
    recordings.loc["A", "leverin"] :, (slice(None), ["x", "y"])
].sort_index(axis=1)
tracking

# %% [markdown]
#
# DLC uses -1 to indicate missing data in this case, which we can
# replace with NaN (not a number).

# %%
tracking = tracking.mask(tracking < 0)

# %% [markdown]
#
# Let's also get the time in seconds from the start of the session.

# %%
tracking["time"] = (
    tracking.index.get_level_values("frame_id") - recordings.loc["A", "leverin"]
) / 30
tracking.set_index("time", append=True, inplace=True)
tracking

# %% [markdown]
#
# We can pull all that together into a function.


# %%
def load_track_session(dlc_path, firston, laston, firstmed):
    """Load one session analysed by DeepLabCut into a DataFrame"""
    try:
        df = pd.read_hdf(dlc_path)
    except FileNotFoundError:
        print(f"File not found: {dlc_path}")
        return None
    df.columns = df.columns.droplevel(0)
    df.index.name = "frame_id"
    df = df.loc[firston:, (slice(None), ["x", "y"])].sort_index(axis=1)
    # Drop the likelihood column and substitute NaN for -1
    df = df.mask(df < 0)
    df["time"] = (df.index.get_level_values("frame_id") - firston) / 30
    df.set_index("time", append=True, inplace=True)
    return df


# %% [markdown]
#
# So now we can just load the tracking data for both acquisitions.

# %%


json_paths = {
    "A": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-A.json"
    ),
    "B": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-B.json"
    ),
}

dlc_paths = {
    "A": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-A_vidcroppedDLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110_filtered.h5"
    ),
    "B": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-B_vidcroppedDLC_HrnetW32_acanJan28shuffle5_detector_360_snapshot_110_filtered.h5"
    ),
}

tracking_dict = {
    acq: load_track_session(
        dlc_path,
        recordings.loc[acq, "firston"],
        recordings.loc[acq, "laston"],
        recordings.loc[acq, "firstmed"],
    )
    for acq, dlc_path in dlc_paths.items()
}
tracking = pd.concat(tracking_dict, names=["acq"])
tracking

# %% [markdown]
#
# At this point we can plot the animal's position over time. Holoviews
# is the library underlying the "hvplot" functionality you saw in the
# first exercise.

# %%
import holoviews as hv
from holoviews import opts

hv.extension("bokeh")
import panel

panel.extension(comms="vscode")
import cv2

# %%
video_path = (
    rawdata_path
    / f"sub-{subject}"
    / f"ses-{session}"
    / f"sub-{subject}_ses-{session}_task-{task}_acq-A_vidcropped.mp4"
)
cap = cv2.VideoCapture(video_path)
for _ in range(recordings.loc["A", "firston"]):
    _, _ = cap.read()
ret, frame = cap.read()
h, w, _ = frame.shape
neckA = tracking.loc[("A"), ("neck", ("x", "y"))].droplevel(0, axis=1)
# For some reason the x-axis is flipped as well as the y axis?
(
    (hv.RGB(frame[::-1, :, ::-1], bounds=(0, 0, w, h)) * hv.Path(neckA)).opts(
        data_aspect=1, frame_width=400
    )
)

# %%

# %% [markdown]
#
# That doesn't really tell us much about the behaviour. Let's try to
# specifically look at the time around the lever presses.
#
# First get a function to load the events sessions (this is basically
# just doing what we did in the first script)


# %%
def load_events_session(events_path, block):
    """Load one session analysed by DeepLabCut into a DataFrame"""
    try:
        df = pd.read_csv(events_path)
    except FileNotFoundError:
        print(f"File not found: {events_path}")
        return None
    df = df.replace({"lp": "llp" if block[0] == "L" else "rlp"})
    df["onset"] = pd.to_timedelta(df["onset"], unit="s")
    return df.fillna(0.01).set_index("onset")


# %%
events_paths = {
    "A": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-A_events.csv"
    ),
    "B": (
        rawdata_path
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_task-{task}_acq-B_events.csv"
    ),
}

events_dict = {
    acq: load_events_session(events_path, recordings.loc[acq, "block"])
    for acq, events_path in events_paths.items()
}
events = pd.concat(events_dict, names=["acq"])
events


# %% [markdown]
#
# We need a function to extract the data around the event times.


# %%
def get_event_windows(df, onset, window_range=(-0.5, 0.0)):
    nearest_i = np.argmin(np.abs(df.index.get_level_values("time") - onset))
    nearest_frame = df.index.get_level_values("frame_id")[nearest_i]
    frame_range = np.rint(np.array(window_range) * 30).astype(int)
    fs, fe = frame_range + nearest_frame
    ev_window = df.loc[(slice(fs, fe), slice(None)), :].copy()
    ev_window["window_offset"] = np.arange(
        frame_range[0], frame_range[1] + 1, dtype=int
    )
    return ev_window.set_index("window_offset")


# %% [markdown]
#
# Let's get the windows for the neck position right before lever press.

# %%
neck = tracking.loc[:, ("neck", ("x", "y"))].droplevel(0, axis=1)
event_cols = ["acq", "onset", "event_id"]
lp_events_df = events.query('event_id.isin(["llp", "rlp"])').reset_index()
lp_events_df["onset"] = lp_events_df["onset"].dt.total_seconds()
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(neck.loc[(x.iloc[0].acq), :], x.iloc[0].onset)
)

# %%
# Let's plot the data around the lever presses
lp_neck_paths = list(
    lp_windows_df.loc[("A")].groupby(["onset"]).apply(lambda x: x.to_numpy())
)
(
    (hv.RGB(frame[::-1, :, ::-1], bounds=(0, 0, w, h)) * hv.Path(lp_neck_paths)).opts(
        data_aspect=1, frame_width=400
    )
)


# %% [markdown]
#
# All the data we just plotted contains absolute position information.
# If we want to look for something in the animal's movement that
# distinguishes between the two lever presses, but is agnostic to
# whether that movement happens to be oriented left or right, we need to
# remove any 'leftness' or 'rightness' to the data. One way we could do
# this is by looking at the absolute _differences_ in the positions,
# i.e. the speed. So let's get the speed of the points of the animal.

# %%
speed = (
    tracking.sort_index()
    .groupby("acq", group_keys=False)
    .apply(
        lambda x: (
            x.diff()
            .pow(2)
            .stack("bodyparts", future_stack=True)
            .sum(skipna=False, axis=1)
            .pow(0.5)
            .unstack("bodyparts")
        )
    )
)
speed

# %% [markdown]
#
# Now we can plot the speed over the session.

# %%
speed.hvplot.line(y="neck", x="time", by="acq", frame_width=600, alpha=0.5)


# %% [markdown]
#
# Now let's get the speed around the lever presses and use those to
# train a classifier.

# %%
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(
        speed.loc[(x.iloc[0].acq), ["neck", "mid_back"]], x.iloc[0].onset
    )
)
lp_beta = lp_windows_df.unstack("window_offset")
lp_beta

# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

clf = LogisticRegressionCV(cv=40, max_iter=10000, class_weight="balanced").fit(
    lp_beta.values, lp_beta.index.get_level_values("event_id")
)


# %% [markdown]
#
# We have now fit a classifier to the data from this day. Let's see how
# well it performs on its own training data (i.e. how well it can
# identify left and right lever presses based only on the speed on the neck).

clf.score(lp_beta.values, lp_beta.index.get_level_values("event_id"))

# %% [markdown]
#
# We can also take a look at _how_ the classifier is making errors, is
# it making more false predictions for left or right lever presses? We
# do this by compariing the true labels and the labels predicted by the
# trained classifer.
pd.DataFrame(
    confusion_matrix(
        lp_beta.index.get_level_values("event_id"),
        clf.predict(lp_beta.values),
        labels=["llp", "rlp"],
    ),
    index=pd.Index(["llp", "rlp"], name="true"),
    columns=pd.Index(["llp", "rlp"], name="predicted"),
)
# %%
