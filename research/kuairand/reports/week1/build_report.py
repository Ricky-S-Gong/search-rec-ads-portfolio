"""Generate a standalone English LaTeX report from checked aggregate evidence.

Final mode refuses incomplete Week 1 stages. Compilation runs in a disposable
working directory and retains only the .tex and final PDF in the output folder.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def escape(value):
    text = str(value)
    return "".join(
        {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }.get(c, c)
        for c in text
    )


def number(value, digits=4):
    return "--" if value is None else f"{value:,.{digits}f}"


def percent(value):
    return "--" if value is None else f"{100 * value:.2f}\\%"


def rows(records):
    return "\n".join(" & ".join(map(str, r)) + r" \\" for r in records)


def metrics_log(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def plot(coordinates, xlabel, ylabel, caption, color="blue"):
    points = " ".join(f"({x},{y:.8g})" for x, y in coordinates)
    return (
        r"""\begin{figure}[H]\centering
\begin{tikzpicture}\begin{axis}[width=.88\linewidth,height=5cm,
  xlabel={"""
        + xlabel
        + r"""},ylabel={"""
        + ylabel
        + r"""},grid=major,
  tick label style={font=\small},label style={font=\small}]
\addplot[color="""
        + color
        + r""",thick] coordinates {"""
        + points
        + r"""};
\end{axis}\end{tikzpicture}\caption{"""
        + caption
        + r"""}\end{figure}"""
    )


def build(
    audit_path,
    stage_path,
    manifest_path,
    output,
    runs=None,
    historical=None,
    allow_incomplete=False,
):
    audit, stage, manifest = map(read, (audit_path, stage_path, manifest_path))
    release = read(Path(stage_path).parent / "release_manifest.json")
    if hashlib.sha256(Path(stage_path).read_bytes()).hexdigest() != release["stage_report_sha256"]:
        raise ValueError("Stage release checksum mismatch")
    if (
        hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        != stage["protocol"]["manifest_sha256"]
    ):
        raise ValueError("Report manifest differs from evaluation")
    if not stage["complete"] and not allow_incomplete:
        raise ValueError(
            "Final report requires a complete stage; use --allow-incomplete only for a marked draft"
        )
    if stage["complete"] and runs is None:
        raise ValueError("Final report requires the actual run logs")
    encoding_path = Path(__file__).parent / "evidence/encoding_audit.json"
    encoding_audit = read(encoding_path) if encoding_path.is_file() else None
    if encoding_audit and encoding_audit["manifest_sha256"] != stage["protocol"]["manifest_sha256"]:
        raise ValueError("Feature encoding audit differs from the report dataset")
    status = "FINAL" if stage["complete"] else "DRAFT -- GPU acceptance pending"
    models = stage["models"]
    splits = audit["splits"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_commit = (
        (Path(runs) / "source_commit.txt").read_text().strip()
        if runs and (Path(runs) / "source_commit.txt").is_file()
        else "cf9f15646934a24312b3a5303ca0785ac5944099"
    )
    if stage["complete"] and (
        len(source_commit) != 40 or not (Path(runs) / "source_commit.txt").is_file()
    ):
        raise ValueError("Final report requires the actual source commit")
    sections = []
    sections.append(
        r"""\begin{titlepage}
\centering\vspace*{1.3cm}
{\Huge\bfseries KuaiRand-Pure\\[.3cm]Week 1 Technical Report\par}
\vspace{.5cm}{\Large Two-Tower Retrieval Baselines and Evaluation\par}
\vspace{.5cm}{\large SAR Recommendation Project\par}
\vspace{.7cm}\fbox{\strut """
        + escape(status)
        + r"""}
\vspace{1cm}
\begin{minipage}{.88\textwidth}
\textbf{Abstract.} This work implements a reproducible long-view retrieval baseline on the official Pure candidate pool. The system checks source identities, removes exact raw duplicates, rebuilds histories using observed playback completion, trains normalized user/item towers, exports aligned vectors and evaluates deterministic exact Top-100 retrieval. The report distinguishes measured results from pending acceptance evidence. Standard-exposure offline metrics describe held-out logged positives; they do not establish an online business improvement.

\medskip\textbf{Current evidence.} The corrected dataset contains """
        + f"{splits['train']['positives']:,}"
        + r""" training positives and """
        + f"{splits['validation']['positives']:,}"
        + r""" validation-positive requests, against """
        + f"{audit['candidate_count']:,}"
        + r""" candidate items. The train-popularity baseline is measured on this protocol. The earlier published enhanced checkpoint is retained as a separate historical diagnostic because its duplicate and history rules differ.

\medskip\textbf{Acceptance status.} """
        + (
            "All three corrected-protocol comparisons and CUDA recovery evidence are present."
            if stage["complete"]
            else "The corrected basic/enhanced CUDA comparison and matching real-GPU interruption/recovery evidence are outstanding. This draft cannot certify Week 1 completion."
        )
        + r"""
\end{minipage}
\vfill{\large Generated from versioned experiment artifacts\par}
\end{titlepage}"""
    )
    sections.append(r"""\section{Problem framing and supervision}
At request time $t$, the system receives an anonymous user identifier, a scenario identifier and previously available positive feedback. It returns up to 100 videos from a time-eligible catalog. This is a \textbf{candidate-retrieval problem}: a user encoder and an item encoder map inputs into a common 64-dimensional space.

\subsection{Target and unit of evaluation}
The target is official $\texttt{long\_view}=1$ in standard, nonrandom exposure. For videos no longer than 18 seconds, the label requires watching at least the video's duration; for longer videos it requires at least 18 seconds. The label is copied from the official logs, not regenerated from our feature snapshot. One positive logged exposure is one evaluation request with one relevant item.

The learned score is normalized dot product,
\[s(u,i)=\langle f_\theta(u,t,h),g_\phi(i)\rangle,\qquad \|f_\theta\|_2=\|g_\phi\|_2=1.\]
It is a ranking score, not a calibrated probability of long viewing. The model learns to rank a positive target above eligible comparison examples, rather than fitting an explicit positive/negative binary classifier on all exposures.

\subsection{Why the baseline has three components}
Popularity measures how far training-period positive frequency alone can go. The basic neural model uses user ID and request context with an ID-only item tower. The enhanced model adds a masked mean of positive history. Both neural runs share data, shuffle seed, output dimension and optimizer budget. Unversioned item content is excluded from the corrected comparison; the earlier content model is documented separately.

\subsection{Interpretation boundary}
Pure contains only logs involving the published candidate pool, so its histories omit other watched videos. Main validation uses standard exposure and can inherit its selection bias. Random exposure is kept separate, and the final test bundle remains sealed. The work does not claim unbiased online retention, CTR or revenue improvement.
\clearpage""")
    data_rows = []
    for split, d in splits.items():
        data_rows.append(
            (
                escape(split),
                f"{d['rows']:,}",
                f"{d['positives']:,}",
                percent(d["positive_rate"]),
                f"{d['users']:,}",
                f"{d['items']:,}",
            )
        )
    sections.append(
        r"""\section{Dataset and exploratory analysis}
The source is the official KuaiRand-Pure archive. Its official MD5 is checked before processing. The shared Drive event/history/feature files are additionally matched to their frozen SHA-256 values. Raw event identities and primary fields are compared to the published event table before any new training artifacts are written.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}\toprule
Window & Exposures & Positives & Positive rate & Users & Items\\\midrule
"""
        + rows(data_rows)
        + r"""
\bottomrule\end{tabular}\caption{Retained nonrandom warmup/train/validation data after exact full-row deduplication. Counts are unique within each window; users/items can overlap across windows.}\end{table}

Training-period item positive counts have median """
        + number(audit["train_item_positive_count_quantiles"][2], 0)
        + r""", 95th percentile """
        + number(audit["train_item_positive_count_quantiles"][4], 0)
        + r""" and maximum """
        + number(audit["train_item_positive_count_quantiles"][-1], 0)
        + r""". The top 100 items account for """
        + percent(audit["train_top100_positive_share"])
        + r""" of training positives. This concentration motivates the popularity comparator.

Training-user exposure counts have median """
        + number(audit["train_user_event_count_quantiles"][2], 0)
        + r""" and 95th percentile """
        + number(audit["train_user_event_count_quantiles"][4], 0)
        + r""". These are exposure counts, not the number of optimizer updates per user.
"""
        + "\nObserved video-duration quantiles (seconds) are: median "
        + number(audit["features"]["duration_ms"]["quantiles"][2] / 1000, 2)
        + ", 75th percentile "
        + number(audit["features"]["duration_ms"]["quantiles"][3] / 1000, 2)
        + ", and 95th percentile "
        + number(audit["features"]["duration_ms"]["quantiles"][4] / 1000, 2)
        + ". The raw duration includes zero values; these are retained and reported, not replaced by a guessed duration.\n"
        + plot(
            list(enumerate(audit["history_length_histogram"])),
            "Available positive-history length",
            "Retained positive requests",
            "History-length distribution across retained warmup/train/validation positives; maximum length 50. Zero-history rate is "
            + percent(audit["empty_history_rate"])
            + ".",
        )
        + r"""
\clearpage"""
    )
    sections.append(
        r"""\section{Preprocessing and temporal audit}
\subsection{Existing preparation}
The publisher maps original anonymous IDs, timestamps, scenario, official label and random flag into standardized events. It assigns time splits and prepares static item metadata plus request-prior positive histories. The earlier consumer fits categorical dictionaries and numeric transformations from training-referenced items rather than the entire catalog.

\subsection{Corrections introduced for this stage}
The new audit removes """
        + f"{audit['exact_duplicates_removed']:,}"
        + r""" rows that are equal across \emph{all} raw CSV columns, while retaining other repeated watches. Request IDs preserve the original filename and row number. No heuristic user-item deduplication is applied.

Histories now require
\[t_{\mathrm{available}}=t_{\mathrm{request}}+\texttt{play\_time\_ms}
<t_{\mathrm{current\ request}}.\]
The inequality is strict: a completion at the same millisecond is unavailable. A delayed completion is inserted in original request order, and the last 50 available positive observations are retained. The completion rule changes """
        + f"{audit['request_prior_histories_changed']:,}"
        + r""" histories relative to request-prior histories on the same deduplicated events. Random observations never become main-training histories.

This is a \textbf{conservative observed-playback proxy}. The source does not provide an independent feedback ingestion timestamp; therefore the audit does not claim exact production feedback availability.

\subsection{Calendar consistency}
Recorded date differs from timestamp-derived dates on """
        + f"{audit['recorded_date_mismatches']['0']:,}"
        + r""" retained exposures under UTC and """
        + f"{audit['recorded_date_mismatches']['8']:,}"
        + r""" under UTC+08. We do not silently declare either source calendar verified. Splits and periodic context instead use the explicit UTC timestamp protocol below.

\begin{center}\begin{tabular}{ll}\toprule
Warmup & $t < $ 2022-04-15 00:00 UTC\\
Train & 2022-04-15 00:00 $\le t <$ 2022-04-29 00:00 UTC\\
Validation & 2022-04-29 00:00 $\le t <$ 2022-05-04 00:00 UTC\\
Test & Sealed; excluded from fitting and this report's EDA\\\bottomrule
\end{tabular}\end{center}
\clearpage"""
    )
    timestamp = int(audit["features"]["user_time_ms"]["examples"][0])
    seconds = timestamp / 1000
    day = (seconds % 86400) / 86400 * 2 * math.pi
    week = ((seconds + 3 * 86400) % (7 * 86400)) / (7 * 86400) * 2 * math.pi
    cyc = [math.sin(day), math.cos(day), math.sin(week), math.cos(week)]
    sections.append(
        r"""\section{User-side features and actual encodings}
\begin{table}[H]\centering\small
\begin{tabularx}{\linewidth}{lXll}\toprule
Feature & Source/meaning & Raw type & Encoded width\\\midrule
User ID & Anonymous training-window ID dictionary & int64 & 32\\
Scenario/tab & Request scenario; valid range 0--14 & int64 & 8\\
Request time & Unix milliseconds; day/week cyclic context & int64 & 4\\
Positive history & At most 50 available prior item IDs & list[int64] & 64 pooled\\
History mask & Valid history slots; padding is excluded & bool[50] & mask only\\\bottomrule
\end{tabularx}\caption{User inputs. Basic omits history; enhanced includes its 64-dimensional masked mean.}\end{table}

User dictionaries are fitted to training-window users only and are numerically sorted. Known IDs map to indices $1,\ldots,V_u$; unseen users use index 0. This reserved embedding has no guaranteed learned cold-start behavior, so unknown-user coverage is reported explicitly. Tab values map to $\texttt{tab}+1$; out-of-range scenarios use reserved index 0.

For timestamp $T$ in milliseconds, with a zero UTC offset,
\begin{align*}
\alpha_d &=2\pi\,(T/1000\bmod86400)/86400,\\
\alpha_w &=2\pi\,((T/1000+3\cdot86400)\bmod604800)/604800.
\end{align*}
The four values are $(\sin\alpha_d,\cos\alpha_d,\sin\alpha_w,\cos\alpha_w)$. The three-day shift aligns the Unix epoch's Thursday with a Monday-based week. An actual retained timestamp, \texttt{"""
        + str(timestamp)
        + r"""}, produces approximately
\["""
        + r",\quad ".join(f"{v:.6f}" for v in cyc)
        + r""".\]

Historical raw item IDs are left-padded with $-1$, distinct from real raw item ID 0. The shared item encoder generates $[B,50,64]$ vectors. The pooled history is
\[h_u={\sum_{\ell=1}^{50}m_\ell g(i_\ell)\over\max(1,\sum_\ell m_\ell)}.\]
An empty history therefore contributes a zero 64-vector and is kept in training/evaluation. Padding vectors never contribute through the mask.
\clearpage"""
    )
    desc = {
        "item_id": "Anonymous candidate ID; train-only embedding dictionary",
        "author_id": "Anonymous video author ID",
        "video_type": "Published video type category",
        "upload_type": "Published upload channel/type",
        "visible_status": "Unversioned visibility snapshot",
        "music_id": "Published music identifier",
        "music_type": "Published music type code",
        "tag": "Published tag string (may contain multiple tags)",
        "video_duration": "Published duration snapshot, milliseconds",
        "server_width": "Published video width, pixels",
        "server_height": "Published video height, pixels",
        "upload_dt": "Upload date, day precision; eligibility proxy",
    }
    feature_rows = []
    for name, meaning in desc.items():
        f = audit["features"][name]
        examples = ", ".join(f["examples"][:2])
        use = (
            "ID embedding"
            if name == "item_id"
            else ("Eligibility only" if name == "upload_dt" else "Excluded")
        )
        feature_rows.append(
            (
                r"\texttt{" + escape(name) + "}",
                escape(meaning),
                escape(examples),
                str(f["cardinality"]),
                str(f["missing"]),
                use,
            )
        )
    raw_item = str(manifest["item_ids"][0])
    index = manifest["item_encoding"]["vocabularies"][0].get(raw_item, 0)
    sections.append(
        r"""\section{Item metadata and feature decisions}
\begingroup\small\setlength{\tabcolsep}{5pt}
\begin{longtable}{p{.16\linewidth}p{.23\linewidth}p{.22\linewidth}rrp{.13\linewidth}}
\toprule Field & Meaning/unit & Actual examples & Values & Nulls & New use\\\midrule\endhead
"""
        + rows(feature_rows)
        + r"""
\bottomrule\end{longtable}\endgroup
The table describes all published item fields even when excluded from the corrected model. Cardinality and null counts are measured on the candidate metadata table, not on test responses. Numeric NaN values count as missing.

\subsection{Corrected formal comparison}
Both item towers use only \texttt{item\_id}. Training-referenced item strings form a sorted dictionary; unseen-in-training candidate categories map to embedding index 0. An actual raw item ID \texttt{"""
        + raw_item
        + r"""} maps to index \texttt{"""
        + str(index)
        + r"""}. Unknown \emph{candidate IDs outside the manifest} are rejected; an unseen training category within the valid candidate catalog is allowed.

Snapshot metadata has no per-request historical version. Its availability cannot be established from this export, so author/type/music/tag/visibility/duration/size fields are excluded from new model inputs. Upload date remains only a conservative next-day candidate-eligibility proxy, with its limitations disclosed.

\subsection{Historical model preprocessing}
The old model uses eight categorical embeddings (8 dimensions each) and four normalized numbers: duration, width, height and relative upload day. It imputes training medians, subtracts training means and divides by training standard deviations; zero standard deviation becomes 1. Unknown categories use 0. Its original metadata buffers and configuration are preserved with the old checkpoint. This historical content model is not renamed as the corrected baseline.
\clearpage"""
    )
    sections.append(r"""\section{Tower architectures}
\subsection{Item tower shared by both corrected models}
\begin{center}\begin{tikzpicture}[node distance=8mm,every node/.style={draw,rounded corners,align=center,minimum width=8cm,minimum height=8mm},>=stealth]
\node (a) {Raw candidate item ID $[N]$};
\node[below=of a] (b) {Frozen training dictionary $\rightarrow$ embedding $[N,8]$};
\node[below=of b] (c) {Linear $8\rightarrow64$, ReLU};
\node[below=of c] (d) {Linear $64\rightarrow64$, L2 normalization};
\draw[->](a)--(b);\draw[->](b)--(c);\draw[->](c)--(d);
\end{tikzpicture}\end{center}

\subsection{User tower}
\begin{center}\begin{tikzpicture}[node distance=8mm,every node/.style={draw,rounded corners,align=center,minimum width=10cm,minimum height=8mm},>=stealth]
\node (a) {User embedding 32 + tab embedding 8 + cyclic time 4};
\node[below=of a] (b) {Basic: 44 inputs\\Enhanced: 44 + masked mean history 64 = 108 inputs};
\node[below=of b] (c) {Linear $44/108\rightarrow128$, ReLU, dropout 0};
\node[below=of c] (d) {Linear $128\rightarrow64$, ReLU, dropout 0};
\node[below=of d] (e) {Linear $64\rightarrow64$, L2 normalization};
\draw[->](a)--(b);\draw[->](b)--(c);\draw[->](c)--(d);\draw[->](d)--(e);
\end{tikzpicture}\end{center}

Batch normalization is disabled. History and targets share item-tower parameters, so gradients from the user-history pathway and positive targets both update the same item encoder. The basic path does not encode unused histories. A singleton final batch is consumed rather than dropped.

The historical item MLP has input width $8\times8+4=68$ and layers $68\rightarrow64\rightarrow64$. Its history-enhanced user width remains 108, but its item representations and source protocol differ.
\clearpage""")
    sections.append(r"""\section{Loss, optimization and training budget}
For a batch of $B$ positive user-item pairs, let $D_i$ contain off-diagonal targets sharing the same item ID as target $i$. Training uses
\[\mathcal L=-{1\over B}\sum_{i=1}^B\log
{\exp(s(u_i,v_i)/\tau)\over\sum_{j\notin D_i}\exp(s(u_i,v_j)/\tau)},\qquad\tau=0.1.\]
The diagonal stays valid. Same-target off-diagonal entries are masked to $-\infty$ before cross-entropy. Different batch targets can still be false negatives: a user may also like them. No claim is made that these are observed dislikes. The initial baseline has no popularity/log-frequency correction for sampled comparisons.

\begin{table}[H]\centering
\begin{tabular}{ll}\toprule Configuration & Corrected basic and enhanced\\\midrule
Output dimension & 64, both towers normalized\\
Seed and shuffle & 42; deterministic global row permutation\\
Optimizer & Adam, $\beta_1=0.9$, $\beta_2=0.999$\\
Learning rate / weight decay & 0.001 / 0\\
Batch size / epochs & 256 / 1 full positive-sample traversal\\
Precision / device & CUDA AMP for the acceptance runs\\
Dropout / batch normalization & 0 / disabled\\
Temperature & 0.1\\
History (enhanced only) & At most 50 available positive items, masked mean\\
Logging / checkpoint & Every 50 / 500 steps, plus epoch-end checkpoint\\\bottomrule
\end{tabular}\caption{Frozen Week 1 training defaults. There is no validation-based hyperparameter search in this one-epoch comparison.}\end{table}

Checkpoint state includes model parameters, Adam state, AMP scaler, Python, NumPy, PyTorch and CUDA RNG states, epoch, next-batch cursor and consumed count. Publication writes the model before a checksummed completion marker. Recovery selects only intact completed checkpoint files and validates binding of code, configuration, data, runtime and device.

Validation in-batch loss depends on batch composition and the available denominator. It is not interchangeable with Recall@100 or NDCG@100, and a smaller last batch can have a mechanically smaller loss.
\clearpage""")
    train_rows = []
    loss_plots = []
    for name in ("basic", "enhanced"):
        if name not in models:
            train_rows.append((name, "Pending", "--", "--", "--", "--"))
            continue
        path = Path(runs) / name / "metrics.jsonl"
        logs = metrics_log(path)
        tr = [r for r in logs if "train/loss" in r]
        va = [r for r in logs if "validation/inbatch_loss" in r]
        if not tr or not va:
            raise ValueError("Missing measured training/validation logs")
        validation = va[-1]["validation/inbatch_loss"]
        steps = len(tr)
        total = sum(r.get("train/step_seconds", 0) + r["train/data_wait_seconds"] for r in tr)
        peak = max((r.get("train/peak_gpu_memory_bytes", 0) for r in tr), default=0)
        train_rows.append(
            (
                name,
                str(models[name]["artifact"]["consumed"]),
                str(steps),
                number(validation),
                number(total, 2),
                number(peak / 2**30, 3),
            )
        )
        loss_plots.append(
            plot(
                [(r["train_step"], r["train/loss"]) for r in tr],
                "Optimizer step",
                "Cross entropy",
                escape(name) + " measured training loss.",
                "blue" if name == "basic" else "orange",
            )
        )
    historical_text = "No historical bundle was supplied to this build."
    if historical:
        h = Path(historical)
        logs = metrics_log(h / "model/metrics.jsonl")
        tr = [r for r in logs if "train/loss" in r]
        va = [r for r in logs if "validation/inbatch_loss" in r]
        publication = read(h / "model/run_manifest.json")
        historical_text = (
            r"The original run \texttt{"
            + escape(publication["run_id"])
            + r"} consumed "
            + f"{publication['consumed']:,}"
            + r" positives in "
            + str(len(tr))
            + r" optimizer steps. Its validation in-batch loss is "
            + number(va[-1]["validation/inbatch_loss"], 6)
            + r". These are historical measurements under the original protocol. Local reproduction verified the frozen source, config, manifest, vector and checkpoint checksums before diagnostic inference."
        )
        loss_plots.append(
            plot(
                [(r["train_step"], r["train/loss"]) for r in tr],
                "Optimizer step",
                "Cross entropy",
                "Historical enhanced run; the final batch has 108 examples rather than 256, explaining part of the last-step drop.",
                "gray",
            )
        )
    sections.append(
        r"""\section{Training results and provenance}
\begin{table}[H]\centering\small\begin{tabular}{lrrrrr}\toprule
Model & Consumed & Steps & Val. loss & Step+wait s & Peak GiB\\\midrule
"""
        + rows(train_rows)
        + r"""\bottomrule\end{tabular}
\caption{Corrected-protocol runs. Step+wait cost excludes checkpoint upload and full validation. Pending cells are not estimates.}\end{table}
"""
        + historical_text
        + "\n"
        + "".join(loss_plots)
        + r"""
The private Drive retains full checkpoint and vector files; W\&B stores run logs. Git retains source, checksums, aggregate evidence and this report. The source commit recorded by the notebook is immutable for a run; branch HEAD changes are not substituted for the actually executed version.
\clearpage"""
    )
    eval_rows = []
    for name in ("popularity", "basic", "enhanced"):
        m = models.get(name, {}).get("metrics", {})
        eval_rows.append(
            (
                name,
                str(m.get("requests", "Pending")),
                number(m.get("recall_at_100"), 6),
                number(m.get("ndcg_at_100"), 6),
                percent(m.get("target_coverage")),
                percent(m.get("insufficient_k_rate")),
            )
        )
    sections.append(
        r"""\section{Evaluation protocol and measured results}
Validation requests are all retained nonrandom positive exposures in the frozen validation window. The same request identities, catalog and eligibility rule apply to all three corrected-protocol systems. Available earlier validation feedback may enter later histories without updating model parameters. Repeat-view candidates are allowed.

Upload dates have day precision. An item becomes eligible at 00:00 UTC on the day after its upload date. Missing eligibility dates are excluded and counted. This is a conservative publication-date proxy; historical removals and visibility changes cannot be reconstructed from the snapshot.

\begin{table}[H]\centering\small\begin{tabular}{lrrrrr}\toprule
Model & Requests & Recall@100 & NDCG@100 & Coverage & Short lists\\\midrule
"""
        + rows(eval_rows)
        + r"""\bottomrule\end{tabular}
\caption{Corrected-protocol exact retrieval. Unavailable targets are retained in the metric denominator.}\end{table}

For one relevant target per request, with rank $r_q$,
\[\operatorname{Recall@100}=\operatorname{HitRate@100}={1\over Q}\sum_q\mathbf1[r_q\le100],\]
\[\operatorname{NDCG@100}={1\over Q}\sum_q{\mathbf1[r_q\le100]\over\log_2(r_q+1)}.\]
A target absent from the eligible candidate set is a miss. Covered-only metrics are secondary diagnostics, not replacements for the end-to-end result. Empty slices use null rather than a fabricated zero.

Scores are computed by bounded request/item matrix blocks. Each block is merged into an exact Top-100 using score-descending, raw-ID-ascending sorting, including boundary ties. Missing result slots are explicit $-1$ sentinels. Vector shards must cover the full manifest catalog exactly, without duplicate IDs, and match the checkpoint's binding and unit norm requirement.
\clearpage"""
    )
    pop = models.get("popularity", {})
    pm = pop.get("metrics", {})
    perf = pop.get("performance", {})
    hist_metric_path = (
        Path(historical).parent / "historical-evaluation.json" if historical else None
    )
    diag = "Historical retrieval was not supplied to this build."
    if hist_metric_path and hist_metric_path.exists():
        hm = read(hist_metric_path)["metrics"]
        diag = (
            "Frozen-source CPU diagnostic inference on the original "
            + f"{hm['requests']:,}"
            + " validation positives yields Recall@100 "
            + number(hm["recall_at_100"], 6)
            + " and NDCG@100 "
            + number(hm["ndcg_at_100"], 6)
            + ". This is not part of the corrected comparison: the old model was trained with original duplicates, request-prior histories and snapshot content. It cannot certify the new protocol or be used to infer a fair improvement over the new popularity baseline."
        )
    sections.append(
        r"""\section{Coverage, performance and reliability}
The corrected validation package contains """
        + percent(pm.get("unknown_user_rate"))
        + r""" unknown-user requests and """
        + percent(pm.get("empty_history_rate"))
        + r""" empty-history requests. The measured candidate target coverage is """
        + percent(pm.get("target_coverage"))
        + r"""; no positive target is silently dropped for unsupported users or missing eligibility.

The local popularity exact-retrieval run uses batches of """
        + str(perf.get("request_batch", "--"))
        + r""" requests after one warmup batch. Its total retrieval cost is """
        + number(perf.get("retrieval_seconds"), 3)
        + r""" seconds, with batch P50 """
        + number(perf.get("p50_batch_seconds"), 4)
        + r""" seconds and batch P95 """
        + number(perf.get("p95_batch_seconds"), 4)
        + r""" seconds. These are observed local CPU batch timings, not production single-request latency. Hardware and timing scope are retained in evaluation JSON; CPU/GPU runs must not be treated as the same performance setting.

\subsection{Historical checkpoint diagnostic}
"""
        + escape(diag)
        + r"""

\subsection{Correctness and acceptance}
Tests cover hand-calculated rank metrics, block/full equivalence, deterministic ties, temporal eligibility, uncovered targets, short lists, duplicate candidate IDs, completion-time histories, same-millisecond requests, unknown users and empty history. Model integration tests verify ID-only feature exclusion, normalized vectors, strict checkpoint binding and deterministic CPU recovery.

Real CUDA acceptance deliberately interrupts the enhanced run at step 25 and restores it. The final model, optimizer, scaler, RNG state and sample cursor must match the uninterrupted reference exactly. CPU correctness does not stand in for this hardware check. The release generator requires both full CUDA model runs plus recovery evidence tied to the exact enhanced checkpoint.

\subsection{Outstanding evidence}
"""
        + escape(
            ", ".join(stage["blockers"])
            if stage["blockers"]
            else "None: the stage acceptance artifact is complete."
        )
        + r"""
\clearpage"""
    )
    sections.append(
        r"""\section{Week 2 work and experiment discipline}
\begin{table}[H]\centering\small
\begin{tabularx}{\linewidth}{lXX}\toprule
Workstream & Next task & Acceptance evidence\\\midrule
Data & Publish audited 27K data and stable nested user subsets; investigate date semantics and metadata provenance & Source/row reconciliation, fixed subset hashes, no future history\\
User model & Compare no-history versus masked mean history with the shared trainer & Same budget, seed, requests and candidate protocol\\
Item model & Admit content only with defensible as-of provenance; compare ID-only versus content & Frozen train-only transformations and explicit feature availability\\
Joint experiments & Run basic, history-only, item-content-only and combined configurations & Four comparable validation results and cost measurements\\
Evaluation & Add long-tail, cold-start and eligibility coverage slices & Frozen definitions, per-slice counts, null for empty groups\\
Scaling & Estimate full traversal and encoding cost from measured throughput & Explicit GPU type, memory and data-wait measurements\\\bottomrule
\end{tabularx}\caption{Week 2 priorities. ANN work follows a validated exact-retrieval reference.}\end{table}

Do not widen the search to multiple model depths, dimensions and losses simultaneously. Keep the full traversal budget fixed for the four principal comparisons. Use validation Recall@100 for later selection, retain NDCG@100 and coverage, and preserve all runs including non-improvements.

The final test bundle remains unused for training, tuning and stage selection. Candidate/request changes are versioned; results across changed protocols are not described as direct model improvements. Learned scores remain retrieval scores until a separate calibration or ranking task is established.

\subsection{Known limitations}
"""
        + r"\begin{enumerate}"
        + "\n"
        + "".join(r"\item " + escape(v) + "\n" for v in audit["limitations"])
        + r"""\item One seed and one epoch provide a baseline, not a stability estimate or evidence of optimal convergence.
\item The item ID embedding cannot supply content generalization for unseen training items; the reserved unknown category is not a trained content representation.
\item Batch comparisons can contain false negatives, and the initial loss has no popularity correction.
\end{enumerate}
\clearpage"""
    )
    sections.append(r"\textbf{Executed source commit:} \path{" + source_commit + "}\n")
    appendix_rows = []
    for name in (
        "user_user_id",
        "user_tab",
        "user_time_ms",
        "long_view",
        "play_time_ms",
        "duration_ms",
    ):
        f = audit["features"][name]
        appendix_rows.append(
            (
                r"\texttt{" + escape(name) + "}",
                escape(", ".join(f["examples"][:1])),
                str(f["cardinality"]),
                number(f.get("min"), 0),
                number(f.get("max"), 0),
            )
        )
    protocol = stage["protocol"]
    sections.append(
        r"""\appendix\section{Feature examples and reproduction contract}
\begin{table}[H]\centering\small\begin{tabularx}{\linewidth}{lXrrr}\toprule
Field & Actual values & Values & Min & Max\\\midrule
"""
        + rows(appendix_rows)
        + r"""\bottomrule\end{tabularx}\caption{Actual retained warmup/train/validation event examples; anonymous public IDs are not personal identities. User-prefixed names disambiguate event fields from item metadata.}\end{table}

Training batches contain $[B]$ tensors for user ID, target item ID, time, scenario and official label; history item IDs and mask are $[B,50]$. The formal ID-only item tower ignores standardized compatibility columns \texttt{category} and \texttt{duration}. Nonrandom positive filtering uses \texttt{long\_view} and \texttt{is\_random}; request ID and split are protocol fields rather than learned features.

\begin{description}
\item[Prepared manifest SHA-256] \hfill\\\path{"""
        + protocol["manifest_sha256"]
        + r"""}
\item[Validation request identity hash] \hfill\\\path{"""
        + protocol["requests_sha256"]
        + r"""}
\item[Candidate artifact hash] \hfill\\\path{"""
        + protocol["candidates_sha256"]
        + r"""}
\item[Official raw archive SHA-256] \hfill\\\path{"""
        + audit["raw_archive_sha256"]
        + r"""}
\end{description}

The team Drive separates raw data, versioned datasets, experiment runs and frozen releases by project and stage. The public notebook is \texttt{week1\_baseline.ipynb}. It reads mounted prepared files directly. Optional copying to \texttt{/content} is a performance cache and does not re-download the source dataset.

Rebuild this report with \texttt{build\_report.py}, providing the audit, matching manifest, stage release and real run-log root. Final generation refuses incomplete stages. Compilation uses an isolated temporary directory; auxiliary files are removed automatically. The standalone source includes its plot data and architecture diagrams.

\begin{thebibliography}{9}
\bibitem{kuairand} C. Gao et al. \emph{KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos}. CIKM 2022, pp. 3953--3957. \url{https://arxiv.org/abs/2208.08696}.
\bibitem{dataset} KuaiRand project. Official dataset definitions and downloads. \url{https://kuairand.com/}.
\end{thebibliography}
"""
    )
    if encoding_audit:
        examples = []
        for entry in encoding_audit["historical_encoding_example"]:
            raw = "Missing" if entry["raw_value"] is None else escape(entry["raw_value"])
            encoded = (
                number(entry["encoded_value"], 6)
                if entry["representation"] == "training-imputed z-score"
                else str(entry["encoded_value"])
            )
            examples.append(
                (escape(entry["feature"]), raw, encoded, escape(entry["representation"]))
            )
        sections.append(r"\clearpage\section{Actual historical feature transformations}" + "\n")
        sections.append(
            "For historical candidate ID "
            + str(encoding_audit["historical_candidate_id"])
            + ", these values reproduce the frozen encoder. Numeric values are checked against the stored training median, mean and standard deviation; no held-out fitting is performed.\n"
        )
        sections.append(
            r"\begin{center}\small\begin{tabularx}{\linewidth}{lrrX}\toprule Feature & Raw value & Encoded value & Representation\\\midrule"
            + "\n"
            + rows(examples)
            + r"\bottomrule\end{tabularx}\end{center}"
            + "\n"
        )
        sections.append(
            "Under the corrected ID-only dictionary, "
            + percent(encoding_audit["unknown_catalog_item_rate"])
            + " of catalog items and "
            + percent(encoding_audit["unknown_validation_item_rate"])
            + " of validation targets map to the reserved unknown item category. These differ from targets outside the eligible catalog. All valid candidate IDs remain in retrieval.\n"
        )
    preamble = (
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,tabularx,longtable,float}
\usepackage{tikz,pgfplots}\usetikzlibrary{positioning}\pgfplotsset{compat=1.18}
\usepackage{xcolor}\definecolor{accent}{HTML}{164B68}
\usepackage{hyperref}\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyhead[L]{KuaiRand-Pure | Week 1}\fancyhead[R]{"""
        + escape(status)
        + r"""}\fancyfoot[C]{\thepage}\setlength{\headheight}{14pt}
\setlength{\parskip}{.5em}\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\begin{document}
"""
    )
    filename = "technical_report" if stage["complete"] else "technical_report_draft"
    tex = output / (filename + ".tex")
    tex.write_text(preamble + "\n".join(sections) + "\n\\end{document}\n")
    evidence = {
        "complete": stage["complete"],
        "audit_sha256": hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
        "stage_sha256": release["stage_report_sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "blockers": stage["blockers"],
        "source_commit": source_commit,
    }
    (output / "report_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return tex


def compile_pdf(tex):
    executable = shutil.which("pdflatex")
    if not executable:
        raise ValueError("pdflatex unavailable; preserve source and use the built-in LaTeX editor")
    with tempfile.TemporaryDirectory(prefix="kuairand-report-") as directory:
        local = Path(directory) / tex.name
        shutil.copy2(tex, local)
        for _ in range(2):
            result = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", local.name],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise ValueError(result.stdout[-6000:])
        overflow = [line for line in result.stdout.splitlines() if "Overfull" in line]
        if overflow:
            print("LaTeX overflow diagnostics:", "\n".join(overflow))
        pdf = tex.with_suffix(".pdf")
        shutil.copy2(local.with_suffix(".pdf"), pdf)
    return pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "stage", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    tex = build(
        args.audit,
        args.stage,
        args.manifest,
        args.output,
        args.runs,
        args.historical,
        args.allow_incomplete,
    )
    print(compile_pdf(tex) if args.compile else tex)
