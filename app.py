"""WiMeP web interface. Run with: streamlit run app.py

Community Cloud gives an app at most about 2.7 GB of memory, so only one species model is kept in memory at a time
and predictions run one at a time across all sessions.
"""
import ctypes, datetime, gc, html, os, threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are bundled; transformers never needs the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import altair as alt
import numpy as np
import streamlit as st

import wimep

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "web", "images")
MAX_BP = wimep.MAX_WEB_BP
GRAY = "#7d8781"
SPECIES_LABELS = {"rufipogon": "*Oryza rufipogon*", "tauschii": "*Aegilops tauschii*", "both": "Both models"}
SPECIES_NOTES = ["Wild rice", "Wild wheat, D-genome donor", "Run both and compare"]
SHORT_NAMES = {"rufipogon": "O. rufipogon", "tauschii": "Ae. tauschii"}

SAMPLES = {
    "O. rufipogon, 41-bp sites": ("rufipogon", """>Oryza_rufipogon_1:7748805:-
GTGTATGATGGGAGGTGGGCCGATTCAGTTTTTTCTACGGG
>Oryza_rufipogon_12:22824527:+
CATAGACCAACGAATTCATTCGAACCATCGGAGCACTGTGA
>Oryza_rufipogon_12:19501887:+
CTGCCGGCACCGGTGCTGGCCGTGCAGGTGACGGAGTTGGC
>Oryza_rufipogon_4:23437147:-
CTCATCACCACCGCCGTCTCCGCCGACCTCGGCACGCACAG
"""),
    "Ae. tauschii, 41-bp sites": ("tauschii", """>Aegilops_tauschii_6D:450103656:-
AAAAATAATGCGGTGCATGTCGATGAAAAAGACAATCAGTG
>Aegilops_tauschii_7D:531560165:+
CCCGATGAGTGCCACGGGATCGTACGTCTGTCGATTCCAGC
>Aegilops_tauschii_3D:423216116:-
GGTCGTGGTGGCGGTGACCACGTCGGCCGCCGTCGGGCTCG
>Aegilops_tauschii_7D:426673219:+
CTGCGGATCGAGATCGTCGCCGGGGACGACGGGCCGGGGGC
"""),
    "O. rufipogon, 600-bp region": ("rufipogon", open(os.path.join(HERE, "example", "Oryza_rufipogon_region.fasta")).read()),
}

PEOPLE = [
    ("Dipro Sinha", "Post Doctoral Fellow", "Department of Veterinary and Biomedical Sciences, South Dakota State University, Brookings, SD 57007, USA", ""),
    ("Subham Ghosh", "Ph.D. Scholar", "Discipline of Bioinformatics, ICAR–Indian Agricultural Research Institute, Pusa, New Delhi 110012, India", ""),
    ("Abhik Sarkar", "Ph.D. Scholar", "Discipline of Bioinformatics, ICAR–Indian Agricultural Research Institute, Pusa, New Delhi 110012, India", ""),
    ("Sougata Bhattacharjee", "Scientist", "School of Crop Sciences, ICAR–Indian Agricultural Research Institute, Jharkhand, Hazaribagh, India", ""),
    ("Sayanti Guha Majumdar", "Scientist", "ICAR–Indian Sugarcane Research Institute, Lucknow, India", "https://scholar.google.com/citations?user=wjAkz68AAAAJ&hl=en"),
    ("Himanshushekhar Chaurasia", "Scientist", "Computer Applications and IT, ICAR-Central Institute for Research on Cotton Technology, India", "https://scholar.google.com/citations?user=4XtHugQAAAAJ&hl=en"),
    ("Sneha Murmu", "Scientist", "Division of Agricultural Bioinformatics, ICAR–Indian Agricultural Statistics Research Institute, Pusa, New Delhi 110012, India", ""),
]


def call_colors():
    """Colours for the 5mC and non-5mC calls, chosen for the active light or dark theme."""
    return ("#e0662e", "#4a8fe0") if st.context.theme.type == "dark" else ("#e0662e", "#2a78d6")


# ---------------------------------------------------------------- prediction
@st.cache_resource
def model_state():
    """Shared across sessions: one lock, and the species currently in memory."""
    return {"lock": threading.Lock(), "loaded": None}


def validate(text):
    df = wimep.parse_fasta(text.strip() or ">")
    df = df[df.sequence != ""]
    if df.empty:
        raise ValueError("No sequence given.")
    short = df[df.sequence.str.len() < 41]
    if len(short):
        names = ", ".join(f"{r.seq_id} ({len(r.sequence)} bp)" for r in short.head(5).itertuples())
        raise ValueError(f"Each sequence must be at least 41 bp; too short: {names}{' ...' if len(short) > 5 else ''}")
    if df.sequence.str.len().sum() > MAX_BP:
        raise ValueError(f"Web input is limited to {MAX_BP:,} bp; use the command-line tool for larger inputs.")
    return df


def predict_one(df, sp):
    """Scores at the default threshold; the results page applies the viewer's threshold afterwards."""
    state = model_state()
    with state["lock"]:
        if state["loaded"] != sp:  # free the other species first, so two are never in memory together
            wimep.load_species.cache_clear()
            state["loaded"] = None
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError:
                pass  # not glibc
            wimep.load_species(sp)
            state["loaded"] = sp
        return wimep.predict(df, sp, "auto", 0.5)


def run_prediction(text, species):
    df = validate(text)  # raises before the status box opens, so input errors stay simple
    species_list = list(wimep.SPECIES if species == "both" else [species])
    results = {}
    with st.status("Running prediction", expanded=True) as status:
        try:
            for sp in species_list:
                if model_state()["lock"].locked():
                    st.write("Waiting for another job to finish.")
                note = "" if model_state()["loaded"] == sp else ", loading the model first"
                st.write(f"Scoring with the *{SHORT_NAMES[sp]}* model{note}.")
                res, mode = predict_one(df, sp)
                if res.empty:
                    raise ValueError("No window could be scored: no 41-bp window has a cytosine (C, or G on the opposite strand) at its centre with only A/C/G/T around it.")
                results[sp] = dict(table=res, mode=mode)
        except Exception:
            status.update(label="Prediction failed", state="error", expanded=False)
            raise
        status.update(label="Prediction complete", state="complete", expanded=False)
    return dict(results=results, sequences=dict(zip(df.seq_id, df.sequence)))


# ---------------------------------------------------------------- charts
def call_scale():
    return alt.Scale(domain=["5mC", "non-5mC"], range=list(call_colors()))


def track(rows, threshold):
    data = rows[["position", "strand", "context", "prob_5mC", "prediction"]]
    bars = alt.Chart(data).mark_rule(strokeWidth=2).encode(
        x=alt.X("position:Q", title="Position (bp)", scale=alt.Scale(zero=False, nice=False)),
        y=alt.Y("prob_5mC:Q", title="5mC probability", scale=alt.Scale(domain=[0, 1])),
        y2=alt.datum(0),
        color=alt.Color("prediction:N", title=None, scale=call_scale(), legend=alt.Legend(orient="top")),
        tooltip=["position", "strand", "context", alt.Tooltip("prob_5mC:Q", format=".3f"), "prediction"],
    ).interactive(bind_y=False)
    cutoff = alt.Chart().mark_rule(strokeDash=[3, 3], color=GRAY).encode(y=alt.datum(threshold))
    return (bars + cutoff).properties(height=280)


def histogram(table):
    return alt.Chart(table[["prob_5mC", "prediction"]]).mark_bar(binSpacing=2, cornerRadiusEnd=3).encode(
        x=alt.X("prob_5mC:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="5mC probability"),
        y=alt.Y("count():Q", title="Cytosines"),
        color=alt.Color("prediction:N", title=None, scale=call_scale(), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("prob_5mC:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="Probability"), "prediction", alt.Tooltip("count():Q", title="Cytosines")],
    ).properties(height=280)


def context_bars(table):
    data = table.groupby(["context", "prediction"]).size().reset_index(name="count")
    return alt.Chart(data).mark_bar(size=26, stroke=None).encode(
        y=alt.Y("context:N", title=None, sort=["CG", "CHG", "CHH"]),
        x=alt.X("count:Q", title="Cytosines"),
        color=alt.Color("prediction:N", title=None, scale=call_scale(), legend=alt.Legend(orient="top")),
        order=alt.Order("prediction:N"),
        tooltip=["context", "prediction", alt.Tooltip("count:Q", title="Cytosines")],
    ).properties(height=280)


def compare_chart(both, threshold):
    meth, unmeth = call_colors()
    scale = alt.Scale(domain=["Both 5mC", "Both non-5mC", "Models disagree"], range=[meth, unmeth, GRAY])
    points = alt.Chart(both).mark_circle(size=44, opacity=0.65).encode(
        x=alt.X("rufipogon:Q", scale=alt.Scale(domain=[0, 1]), title="O. rufipogon model probability"),
        y=alt.Y("tauschii:Q", scale=alt.Scale(domain=[0, 1]), title="Ae. tauschii model probability"),
        color=alt.Color("agreement:N", title=None, scale=scale, legend=alt.Legend(orient="top")),
        tooltip=["seq_id", "position", "strand", "context", alt.Tooltip("rufipogon:Q", title="O. rufipogon", format=".3f"), alt.Tooltip("tauschii:Q", title="Ae. tauschii", format=".3f"), "agreement"],
    ).interactive()
    vertical = alt.Chart().mark_rule(strokeDash=[3, 3], color=GRAY).encode(x=alt.datum(threshold))
    horizontal = alt.Chart().mark_rule(strokeDash=[3, 3], color=GRAY).encode(y=alt.datum(threshold))
    return (points + vertical + horizontal).properties(height=460)


def sequence_view(seq, rows, threshold):
    """Sequence in 60-bp lines; scored cytosines are coloured by call."""
    meth, unmeth = call_colors()
    calls = {r.position: ("plus" if r.strand == "+" else "minus") if r.prob_5mC >= threshold else "no" for r in rows.itertuples()}
    styles = {"plus": f"background:{meth};color:#fff;border-radius:2px", "minus": f"outline:1.5px solid {meth};outline-offset:-1px;border-radius:2px", "no": f"color:{unmeth}"}
    lines = []
    for i in range(0, len(seq), 60):
        cells = "".join(
            f'<span style="{styles[calls[j + 1]]}">{html.escape(seq[j])}</span>' if j + 1 in calls else html.escape(seq[j])
            for j in range(i, min(i + 60, len(seq)))
        )
        lines.append(f'<span style="opacity:.55">{i + 1:>6}  </span>{cells}')
    return (f'<p style="font-size:13px;margin:0 0 10px"><span style="{styles["plus"]};padding:0 4px">C</span> 5mC on + strand &nbsp; '
            f'<span style="{styles["minus"]};padding:0 4px">G</span> 5mC on − strand &nbsp; '
            f'<span style="{styles["no"]}">C/G</span> scored, non-5mC</p>'
            '<pre style="font:13px/1.6 monospace;overflow-x:auto;white-space:pre;margin:0">' + "\n".join(lines) + "</pre>")


# ---------------------------------------------------------------- results
def table_tab(table, sp):
    a, b, c = st.columns(3)
    contexts = a.pills("Context", ["CG", "CHG", "CHH"], selection_mode="multi", default=["CG", "CHG", "CHH"], key=f"ctx-{sp}")
    strands = b.pills("Strand", ["+", "-"], selection_mode="multi", default=["+", "-"], key=f"strand-{sp}")
    calls = c.pills("Call", ["5mC", "non-5mC"], selection_mode="multi", default=["5mC", "non-5mC"], key=f"call-{sp}")
    view = table[table.context.isin(contexts) & table.strand.isin(strands) & table.prediction.isin(calls)]
    models = {m: st.column_config.NumberColumn(format="%.3f") for m in wimep.MODELS}
    st.dataframe(
        view, hide_index=True, height=440,
        column_order=["seq_id", "position", "strand", "context", "prob_5mC", "prediction", "window", *wimep.MODELS, "window_start", "window_end"],
        column_config={"prob_5mC": st.column_config.ProgressColumn("prob_5mC", format="%.3f", min_value=0, max_value=1), "seq_id": st.column_config.TextColumn(width="medium"), **models},
    )
    st.download_button(
        f"Download {len(view):,} rows (CSV)", view.to_csv(index=False), f"WiMeP_{sp}_{datetime.date.today()}.csv", "text/csv",
        key=f"csv-{sp}", icon=":material/download:", on_click="ignore",
    )


def species_results(sp, res, sequences, threshold):
    table = res["table"].assign(prediction=lambda d: np.where(d.prob_5mC >= threshold, "5mC", "non-5mC"))
    n_meth, ids = int((table.prediction == "5mC").sum()), table.seq_id.unique()
    a, b, c, d = st.columns(4)
    a.metric("Cytosines scored", f"{len(table):,}", border=True)
    b.metric(f"Predicted 5mC (≥ {threshold:g})", f"{n_meth:,}", border=True)
    c.metric("Share predicted 5mC", f"{n_meth / len(table):.1%}", border=True)
    d.metric("Sequences", f"{len(ids):,}", border=True)

    scan = res["mode"] == "scan"
    names = (["Probability track", "Sequence view"] if scan else []) + ["Table", "Summary"]
    tabs = dict(zip(names, st.tabs(names)))
    if scan:
        seq_id = ids[0]
        if len(ids) > 1:
            with tabs["Probability track"]:
                seq_id = st.selectbox("Sequence", ids, key=f"seq-{sp}")
        rows = table[table.seq_id == seq_id]
        with tabs["Probability track"]:
            st.altair_chart(track(rows, threshold), width="stretch")
            st.caption(f"{seq_id}: each line is one scored cytosine on either strand; the dashed line is the threshold. Scroll to zoom and drag to pan.")
        with tabs["Sequence view"]:
            st.caption(str(seq_id))
            st.html(sequence_view(sequences[seq_id], rows, threshold))
    with tabs["Table"]:
        table_tab(table, sp)
    with tabs["Summary"]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Probability distribution**")
            st.altair_chart(histogram(table), width="stretch")
        with right:
            st.markdown("**Calls by sequence context**")
            st.altair_chart(context_bars(table), width="stretch")


def compare_results(results, threshold):
    keys = ["seq_id", "position", "strand", "context"]
    both = results["rufipogon"]["table"][keys + ["prob_5mC"]].rename(columns={"prob_5mC": "rufipogon"}).merge(
        results["tauschii"]["table"][keys + ["prob_5mC"]].rename(columns={"prob_5mC": "tauschii"}), on=keys)
    r, t = both["rufipogon"] >= threshold, both["tauschii"] >= threshold
    both["agreement"] = np.select([r & t, ~r & ~t], ["Both 5mC", "Both non-5mC"], "Models disagree")
    a, b, c = st.columns(3)
    a.metric("Calls in agreement", f"{(r == t).mean():.1%}", border=True)
    b.metric("5mC only in *O. rufipogon*", f"{(r & ~t).sum():,}", border=True)
    c.metric("5mC only in *Ae. tauschii*", f"{(t & ~r).sum():,}", border=True)
    st.altair_chart(compare_chart(both, threshold), width="stretch")
    st.caption("Each point is one cytosine scored by both models; dashed lines mark the threshold. Scroll to zoom and drag to pan.")


@st.fragment
def show_results():
    """A fragment, so that moving the threshold or a filter redraws the results without rerunning the whole page."""
    out = st.session_state.output
    st.divider()
    head, control = st.columns([3, 2], vertical_alignment="center")
    head.subheader("Results")
    threshold = control.slider("Probability threshold for calling 5mC", 0.05, 0.95, 0.5, 0.05, key="threshold")
    results = out["results"]
    if len(results) == 1:
        sp, res = next(iter(results.items()))
        st.markdown(f"**Model:** {SPECIES_LABELS[sp]}")
        species_results(sp, res, out["sequences"], threshold)
    else:
        tabs = st.tabs([SHORT_NAMES[sp] for sp in results] + ["Compare models"])
        for tab, (sp, res) in zip(tabs, results.items()):
            with tab:
                species_results(sp, res, out["sequences"], threshold)
        with tabs[-1]:
            compare_results(results, threshold)


# ---------------------------------------------------------------- pages
def home():
    st.title("WiMeP")
    st.markdown("##### Wild Methylation Predictor")
    st.markdown("An ensemble deep-learning and machine-learning predictor of DNA 5-methylcytosine (5mC) sites in the wild rice *Oryza rufipogon* and the wild wheat progenitor *Aegilops tauschii*.")
    a, b, _ = st.columns([1, 1, 3])
    if a.button("Start prediction", type="primary", icon=":material/arrow_forward:", width="stretch"):
        st.switch_page(PAGE_PREDICT)
    if b.button("Data & downloads", icon=":material/download:", width="stretch"):
        st.switch_page(PAGE_RESOURCES)

    st.space("small")
    a, b, c, d = st.columns(4)
    a.metric("Species models", "2", border=True)
    b.metric("Input window", "41 bp", border=True)
    c.metric("Contexts: CG, CHG, CHH", "3", border=True)
    d.metric("Training sites", "154,640", border=True)

    st.subheader("How to use WiMeP")
    steps = [
        (":material/edit_note:", "1. Prepare input", "FASTA or plain sequence (A, C, G, T) of **at least 41 bp**. A 41-bp sequence with C at position 21 is scored as one site. Longer sequences are cut into overlapping 41-bp windows with a step of 1 bp, and every window whose centre is a cytosine (C, or G for the opposite strand) is scored."),
        (":material/grass:", "2. Choose the model", "*O. rufipogon*, *Ae. tauschii* or both. A separate model is trained for each species, so use the one matching your sequence."),
        (":material/query_stats:", "3. Run prediction", "Results show each scored cytosine's position, strand, context, window coordinates, the probability from each model and the ensemble call. Longer sequences also get a probability track and a sequence view. Download the results as CSV."),
    ]
    for col, (icon, title, text) in zip(st.columns(3), steps):
        with col.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{title}**")
            st.markdown(text)

    st.subheader("Model pipeline")
    with st.container(border=True):
        st.image(os.path.join(IMAGES, "pipeline.svg"), width="stretch")

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.subheader("Background")
        st.markdown("DNA methylation is a heritable epigenetic mark in which a methyl group is added to the fifth carbon of cytosine (5mC). In plants it occurs in three sequence contexts (CG, CHG and CHH, where H = A, C or T) and regulates transposon silencing, gene expression and genome stability.")
        st.markdown("Crop wild relatives such as *O. rufipogon* (the progenitor of cultivated rice) and *Ae. tauschii* (the D-genome donor of bread wheat) are reservoirs of adaptive variation, but their methylomes are far less studied than those of model species. Bisulfite sequencing is costly, so sequence-based prediction offers a fast first look at likely methylated sites.")
    with right.container(border=True, height="stretch"):
        st.subheader("About WiMeP")
        st.markdown("WiMeP predicts whether a cytosine is methylated (5mC; CG, CHG and CHH treated as one class) from the 41-bp sequence centred on it. A separate model is trained for each species.")
        st.markdown("Each model is an optimised ensemble of three machine-learning classifiers on selected handcrafted and DNABERT features (selected by mutual information and stepwise regression), a fine-tuned DNABERT-6 transformer and a DNABERT-BiLSTM. Ensemble weights are fitted on a cluster-disjoint validation set.")
        st.caption("Positive sites come from whole-genome bisulfite sequencing; training and test sets are separated with CD-HIT-EST at 80% identity. The Random Forest received an ensemble weight of effectively zero and is omitted from this hosted version to save memory; it does not change the reported probabilities.")


def load_sample():
    name = st.session_state.sample_pick
    if name:
        st.session_state.species, st.session_state.seq = SAMPLES[name]
        st.session_state.source = "Paste"
        st.session_state.pop("output", None)
    st.session_state.sample_pick = None


def clear_form():
    st.session_state.seq = ""
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.session_state.pop("output", None)


def input_summary(text):
    """One line under the input, so limits are visible before a job is submitted."""
    if not text.strip():
        return f"A, C, G, T only. Each sequence at least 41 bp; up to {MAX_BP:,} bp per job."
    df = wimep.parse_fasta(text)
    return f"{len(df):,} sequence{'s' if len(df) != 1 else ''}, {df.sequence.str.len().sum():,} of {MAX_BP:,} bp allowed."


def predict_page():
    st.title("5mC prediction")
    st.markdown("A 41-bp sequence with C at position 21 is scored as one site. Any other sequence of at least 41 bp is cut into overlapping 41-bp windows, and every window centred on a cytosine is scored.")
    st.session_state.setdefault("seq", "")
    st.session_state.setdefault("species", "rufipogon")
    st.session_state.setdefault("source", "Paste")

    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True, height="stretch"):
        source = st.segmented_control("Input", ["Paste", "Upload"], key="source", label_visibility="collapsed")
        text = ""
        if source == "Upload":
            upload = st.file_uploader("FASTA file", type=["fasta", "fa", "fna", "fas", "txt"], key=f"upload-{st.session_state.get('upload_key', 0)}")
            if upload is not None:
                text = upload.getvalue().decode("utf-8", "replace")
        else:
            text = st.text_area("FASTA or raw sequence", key="seq", height=200, placeholder=">my_sequence\nACGTTGCCGGAGCTCGACGG...")
        st.caption(input_summary(text))
        st.pills("Try a sample", list(SAMPLES), key="sample_pick", on_change=load_sample)
    with right.container(border=True, height="stretch"):
        st.radio("Reference model", list(SPECIES_LABELS), format_func=SPECIES_LABELS.get, captions=SPECIES_NOTES, key="species")
        run = st.button("Run prediction", type="primary", icon=":material/play_arrow:", width="stretch")
        st.button("Clear", on_click=clear_form, width="stretch")

    if run:
        st.session_state.pop("output", None)
        if not text.strip():
            st.error("Choose a FASTA file first." if source == "Upload" else "Paste a sequence first.", icon=":material/error:")
        else:
            try:
                st.session_state.output = run_prediction(text, st.session_state.species)
                st.toast("Prediction complete", icon=":material/check_circle:")
            except ValueError as e:
                st.error(str(e), icon=":material/error:")
            except (OSError, RuntimeError) as e:
                st.error(f"The models could not be loaded: {e}", icon=":material/error:")
    if "output" in st.session_state:
        show_results()


def resources():
    st.title("Resources")
    st.markdown("Example data, the training and testing datasets, and the command-line version of WiMeP.")
    downloads = os.path.join(HERE, "web", "downloads")
    wimep.ensure_files(downloads)
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.subheader("Example data")
        st.markdown("Example inputs with their WiMeP output: a 600-bp *O. rufipogon* genomic region (scan mode) and 41-bp test sites for both species (site mode).")
        with open(os.path.join(downloads, "WiMeP_example_data.zip"), "rb") as f:
            st.download_button("Example data (zip, 95 KB)", f, "WiMeP_example_data.zip", "application/zip", icon=":material/download:", on_click="ignore")
    with right.container(border=True, height="stretch"):
        st.subheader("Training and testing datasets")
        st.markdown("The four 41-bp datasets (5mC positive and negative for each species), split into training and independent test sets, in FASTA format.")
        with open(os.path.join(downloads, "WiMeP_5mC_datasets.zip"), "rb") as f:
            st.download_button("Datasets (zip, 4.3 MB)", f, "WiMeP_5mC_datasets.zip", "application/zip", icon=":material/download:", on_click="ignore")
    st.subheader("Dataset summary")
    st.markdown("""
| Species | Class | Training | Test | Source |
|---|---|--:|--:|---|
| *O. rufipogon* | 5mC (positive) | 42,578 | 9,868 | Whole-genome bisulfite sequencing (2 samples) |
| *O. rufipogon* | non-5mC (negative) | 42,578 | 9,868 | CDS windows, C at centre, CHG motif, < 60% identity |
| *Ae. tauschii* | 5mC (positive) | 34,742 | 8,041 | Bisulfite sequencing (2 samples) |
| *Ae. tauschii* | non-5mC (negative) | 34,742 | 8,041 | CDS windows, C at centre, CHG motif, < 60% identity |
""")
    st.caption("5mC sites: binomial test p < 0.01, methylation level ≥ 0.5, coverage ≥ 5; CG, CHG and CHH merged. Redundancy removed with CD-HIT-EST (90%); train/test split cluster-disjoint at 80% identity. Negative windows have no methylated read at the centre cytosine in the bisulfite data.")
    st.subheader("Command-line tool")
    st.markdown("For genomes or large sequence sets, run WiMeP locally (CPU, CUDA or Apple GPU). Source code and models: [github.com/diprosinha/WiMeP](https://github.com/diprosinha/WiMeP).")
    st.code("""git lfs install
git clone https://github.com/diprosinha/WiMeP.git
cd WiMeP
conda env create -f environment.yml && conda activate wimep
python wimep.py predict --species rufipogon --input my_seqs.fasta --out predictions.csv""", language="bash")


def developers():
    st.title("Developers")
    st.markdown("WiMeP was developed by:")
    columns = st.columns(3)
    for i, (name, role, affiliation, scholar) in enumerate(PEOPLE):
        parts = name.split()
        initials = parts[0][0] + parts[-1][0]
        with columns[i % 3].container(border=True):
            st.html(f'<div style="width:44px;height:44px;border-radius:50%;background:#1f5e4a;color:#f4f1e8;display:flex;align-items:center;justify-content:center;font-weight:700" aria-hidden="true">{initials}</div>')
            st.markdown(f"**{name}**")
            st.caption(role)
            st.markdown(affiliation)
            if scholar:
                st.link_button("Google Scholar", scholar, icon=":material/school:")


st.set_page_config(page_title="WiMeP", page_icon=os.path.join(IMAGES, "mark.svg"), layout="wide")
st.logo(os.path.join(IMAGES, "logo.svg"), size="large", icon_image=os.path.join(IMAGES, "logo.svg"))
# Same arrangement as the original site: logo on the left, menu on the right. Streamlit has no option for this.
st.html("""<style>
[data-testid="stHeader"] { background: #1f5e4a !important; color: #f4f1e8 !important; height: 60px; }
[data-testid="stToolbar"] { height: 60px; }
[data-testid="stHeader"] * { color: #f4f1e8 !important; }
[data-testid="stHeader"] button svg { fill: currentColor; }
[data-testid="stTopNav"] { justify-content: flex-end; }
div:has(> [data-testid="stMainMenu"]) { width: auto; min-width: 0; flex: 0 0 auto; }
[data-testid="stTopNavLink"] { background: transparent !important; border-radius: 8px; padding: 8px 14px; opacity: .82; }
[data-testid="stTopNavLink"] * { color: #f4f1e8 !important; font-size: 15px; font-weight: 500; }
[data-testid="stTopNavLink"]:hover { background: rgba(255, 255, 255, .08) !important; opacity: 1; }
[data-testid="stTopNavLink"][aria-current="page"] { background: rgba(255, 255, 255, .16) !important; opacity: 1; }
</style>""")
PAGE_PREDICT = st.Page(predict_page, title="Predict", url_path="predict")
PAGE_RESOURCES = st.Page(resources, title="Resources", url_path="resources")
page = st.navigation([
    st.Page(home, title="Home", default=True),
    PAGE_PREDICT,
    PAGE_RESOURCES,
    st.Page(developers, title="Developers", url_path="developers"),
], position="top")
page.run()
st.divider()
st.caption("WiMeP · DNA 5mC prediction for crop wild relatives · [Source code](https://github.com/diprosinha/WiMeP) .")
st.caption("Sinha D, Ghosh S, Sarkar A, Bhattacharjee S, Guha Majumdar S, Chaurasia H, Murmu S")
