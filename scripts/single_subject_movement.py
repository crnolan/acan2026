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


# %% [markdown]
#
# Now we can load up some metadata about our acquisitions.

# %%
json_path = get_file_path(subject, session, task, acq, ".json")
with open(json_path) as f:
    recording_info = pd.Series(json.load(f))
recording_info

# %% [markdown]
#
# We can read in a DeepLabCut tracking file (HDF5 format) using the
# pandas read_hdf function.

# %%
dlc_path = get_file_path(
    subject,
    session,
    task,
    acq,
    "DLC_HrnetW32_medass_topviewmouseAug7shuffle3_detector_best-170_snapshot_best-170.h5",
)
tracking = pd.read_hdf(dlc_path)
tracking

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
# The frames are numbered, but we care more about time. The frame times
# are provided in a separate CSV file, so let's read that and use it.

# %%
frame_times_path = get_file_path(subject, session, task, acq, "_sync.csv")
frame_times = pd.read_csv(frame_times_path, header=None, names=["time"])
tracking.index = pd.to_timedelta(frame_times["time"], unit="s")
tracking

# %% [markdown]
#
# These frame times are relative to the start of the video, not the
# start of the session. That doesn't matter for now, but spoiler alert:
# we might need to consider movement in relation to the med events
# later. So let's adjust these frame times to be relative to the start
# of the session as per the events files. The start frame of the session
# is given in the "leverin" column of the recordings table we loaded
# above.

# %%
tracking.index = tracking.index - tracking.index[recording_info.loc["leverin"] - 1]
tracking = tracking.loc[pd.Timedelta(0, unit="s") :]
tracking

# %% [markdown]
#
# To make things simple for now, let's only look at the neck position of
# the animal.

# %%
neck = tracking.loc[:, ("neck")]
neck

# %% [markdown]
#
# At this point we can plot the animal's position over time. Holoviews
# is the library underlying the "hvplot" functionality you saw in the
# first exercise - we've imported holoviews into the "hv" namespace.
# You can see the kinds of plots you can generate with holoviews here:
# https://holoviews.org/gallery/


# %%
video_path = get_file_path(subject, session, task, acq, ".mp4")
cap = cv2.VideoCapture(video_path)
for _ in range(recording_info.loc["leverin"]):
    _, _ = cap.read()
ret, frame = cap.read()
h, w, _ = frame.shape
# neckA = tracking.loc[("A"), ("neck", ("x", "y"))].droplevel(0, axis=1)
# For some reason the x-axis is flipped as well as the y axis?
(
    (
        hv.RGB(frame[::-1, :, ::-1], bounds=(0, 0, w, h))
        * hv.Path(neck.loc[:, ("x", "y")])
    ).opts(data_aspect=1, frame_width=400)
)

# %% [markdown]
#
# This shows us the location of the animal's neck over the session, but
# likely you'll notice some large jumps in the position. These are
# coming from frames where the animal tracking was unsuccessful (DLC
# sets these values to -1). We can get rid of these, and also smooth the
# data by removing points that deeplabcut was not confident about and
# subsequently applying a median filter (and discard the likelihood
# column since we won't be using it anymore).

# %%
neck = (
    neck.mask(neck["likelihood"] < 0.6)
    .rolling(window=pd.Timedelta(seconds=0.2), min_periods=1)
    .median()
    .drop(columns="likelihood")
)

(
    (
        hv.RGB(frame[::-1, :, ::-1], bounds=(0, 0, w, h))
        * hv.Path(neck.loc[:, ("x", "y")])
    ).opts(data_aspect=1, frame_width=400)
)

# %% [markdown]
#
# We can pull all that together into a function.


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
    df = df.loc[:, ("neck")]
    df = (
        df.mask(df["likelihood"] < 0.6)
        .rolling(window=pd.Timedelta(seconds=0.2), min_periods=1)
        .median()
        .drop(columns="likelihood")
    )
    return df


# %% [markdown]
#
# So now we can just load the tracking data for both acquisitions.

# %%
tracking_dict = {
    acq: load_track_session(subject, session, task, acq) for acq in ["A", "B"]
}
neck = pd.concat(tracking_dict, names=["acq"])
neck


# %% [markdown]
#
# That doesn't really tell us much about the behaviour. Let's try to
# specifically look at the time around the lever presses.
#
# First get a function to load the events sessions (this is basically
# just doing what we did in the first script)


# %%
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


# %%
events_dict = {
    acq: load_events_session(subject, session, task, acq) for acq in ["A", "B"]
}
events = pd.concat(events_dict, names=["acq"])
events


# %% [markdown]
#
# We need a function to extract the data around the event times.


# %%
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


# %% [markdown]
#
# Let's get the windows for the neck position right before lever press.

# %%
event_cols = ["acq", "onset", "event_id"]
lp_events_df = events.query('event_id.isin(["llp", "rlp"])').reset_index()
# lp_events_df["onset"] = lp_events_df["onset"].dt.total_seconds()
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
    neck.sort_index()
    .groupby("acq", group_keys=False)
    .apply(lambda x: (x.diff().pow(2).sum(skipna=False, axis=1).pow(0.5)))
    .rename("speed")
)
speed

# %% [markdown]
#
# Now we can plot the speed over the session.

# %%
speed.hvplot.line(y="speed", x="time", by="acq", frame_width=600, alpha=0.5)


# %% [markdown]
#
# Now let's get the speed around the lever presses and use those to
# train a classifier.

# %%
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(speed.loc[(x.iloc[0].acq)].to_frame(), x.iloc[0].onset)
)
lp_beta = lp_windows_df.unstack("window_offset").dropna()
lp_beta

# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

clf = LogisticRegressionCV(
    cv=40,
    max_iter=10000,
    class_weight="balanced",
    l1_ratios=(0,),
    use_legacy_attributes=False,
).fit(lp_beta.values, lp_beta.index.get_level_values("event_id"))


# %% [markdown]
#
# We have now fit a classifier to the data from this day. Let's see how
# well it performs on its own training data (i.e. how well it can
# identify left and right lever presses based only on the speed on the neck).

# %%
clf.score(lp_beta.values, lp_beta.index.get_level_values("event_id"))

# %% [markdown]
#
# We can also take a look at _how_ the classifier is making errors, is
# it making more false predictions for left or right lever presses? We
# do this by comparing the true labels and the labels predicted by the
# trained classifier.

# %%
pd.DataFrame(
    confusion_matrix(
        lp_beta.index.get_level_values("event_id"),
        clf.predict(lp_beta.values),
        labels=["llp", "rlp"],
    ),
    index=pd.Index(["llp", "rlp"], name="true"),
    columns=pd.Index(["llp", "rlp"], name="predicted"),
)

# %% [markdown]
# We can now see how well our classifier does across sessions. The code
# below demonstrates how we can use our existing functions to get the data
# for all sessions.

# %%
sessions = [
    ("RR20prerev.02", "RR20prerev"),
    ("RR20prerev.03", "RR20prerev"),
    ("RR20rev.01", "RR20rev"),
]

# %%
tracking_dict = {
    (session, acq): load_track_session(subject, session, task, acq)
    for session, task in sessions
    for acq in ["A", "B"]
}
neck = pd.concat(tracking_dict, names=["session", "acq"])
neck

# %%
events_dict = {
    (session, acq): load_events_session(subject, session, task, acq)
    for session, task in sessions
    for acq in ["A", "B"]
}
events = pd.concat(events_dict, names=["session", "acq"])
events

# %%
speed = (
    neck.sort_index()
    .groupby(["session", "acq"], group_keys=False)
    .apply(lambda x: (x.diff().pow(2).sum(skipna=False, axis=1).pow(0.5)))
    .rename("speed")
)
speed

# %%
event_cols = ["session", "acq", "onset", "event_id"]
lp_events_df = events.query('event_id.isin(["llp", "rlp"])').reset_index()
lp_windows_df = lp_events_df.groupby(event_cols)[event_cols].apply(
    lambda x: get_event_windows(speed.loc[(x.iloc[0].session, x.iloc[0].acq)].to_frame(), x.iloc[0].onset)
)
lp_beta = lp_windows_df.unstack("window_offset").dropna()
lp_beta

# %% [markdown]
#
# We can now easily select out data from a specific session as
# demonstrated below. Can you score the results of other session using
# the classifier we've already built? Hint: you don't need to create a
# new LogisticRegressionCV object, you can just use the "score" method
# of the existing one.

# %%
lp_beta.loc[("RR20rev.01")]

# %%
