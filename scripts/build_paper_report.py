"""Build paper-ready Markdown, LaTeX and figures from measured benchmark JSON/CSV."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    folders=[ROOT/"outputs/paper_benchmark",ROOT/"outputs/paper_benchmark_threads1"]
    reports=[json.loads((folder/"results.json").read_text()) for folder in folders]
    out=ROOT/"docs/paper_performance"
    out.mkdir(parents=True,exist_ok=True)
    table=[]
    for label,r in zip(["Default CPU","One-thread CPU"],reports):
        s=r["stages"]
        table.append([label,str(r["environment"]["torch_threads"]),f"{r['core_fps']:.2f}",f"{r['local_fps']:.2f}",
                      f"{s['local_frame_ms']['mean_ms']:.2f}",f"{s['local_frame_ms']['p95_ms']:.2f}",
                      f"{r['local_frames_within_source_deadline_percent']:.1f}"])
    # Choose fastest measured configuration but disclose both, including all repetitions.
    best_index=max(range(len(reports)),key=lambda i:reports[i]["local_fps"])
    r=reports[best_index]; s=r["stages"]; v=r["video"]; e=r["environment"]
    meets=r["local_fps"]>=v["source_fps"]
    verdict=("Average processing throughput exceeds the source frame rate in this configuration; this supports a soft, average-throughput real-time claim for this clip and hardware, not hard real-time or camera-to-screen latency."
             if meets else "Neither tested configuration sustained the source frame rate. The measurements establish recorded-video motion-processing performance on this hardware, but do not substantiate full-frame-rate real-time operation.")
    heading=["Configuration","Torch threads","Core FPS","Local FPS","Mean local ms","P95 local ms","Within source deadline (%)"]
    markdown="| "+" | ".join(heading)+" |\n|"+" --- |"*len(heading)+"\n"
    markdown+="\n".join("| "+" | ".join(row)+" |" for row in table)
    paragraph=(f"Computational performance was evaluated using a {v['width']} x {v['height']} video containing "
        f"{r['runs'][0]['frames']} frames at {v['source_fps']:.2f} FPS on an Intel Core i5-1135G7 CPU "
        f"(4 cores, 8 logical processors; approximately 8 GB RAM). CPU FP32 inference used batch size one and "
        f"an image-size setting of 640. After 20 warm-up iterations, three complete passes were timed using a monotonic "
        f"high-resolution clock. The evaluated pipeline comprised YOLO detection, Kalman tracking, independent "
        f"second-order seven-sample polynomial filtering of tracked X and Y coordinates, and first- and second-order "
        f"derivatives for velocity and acceleration. The fastest tested configuration used {e['torch_threads']} inference "
        f"thread(s), yielding {r['core_fps']:.2f} FPS for detection through motion estimation and {r['local_fps']:.2f} FPS "
        f"when decoding, annotation, JPEG encoding and local loop overhead were included. Mean and 95th-percentile "
        f"local frame service times were {s['local_frame_ms']['mean_ms']:.2f} and {s['local_frame_ms']['p95_ms']:.2f} ms, "
        f"respectively, compared with a source-frame budget of {r['source_deadline_ms']:.2f} ms. "
        f"{verdict} Browser rendering, network transfer and camera buffering were excluded.")
    lines=["# Paper performance evidence","",verdict,"","## Ready-to-insert subsection (Section IV.D)","",paragraph,"",
        "## Measured configurations","",markdown,"","Core FPS = 1000 / mean detection-through-motion time. Local FPS = total frames / total measured loop wall time. Do not average instantaneous per-frame FPS.",
        "","## Stage timings for fastest tested configuration","","| Stage | Mean ms | P95 ms |","| --- | ---: | ---: |"]
    for key in ["decode_ms","detection_ms","tracking_ms","motion_ms","drawing_ms","jpeg_ms","core_ms","local_frame_ms"]:
        lines.append(f"| {key} | {s[key]['mean_ms']:.3f} | {s[key]['p95_ms']:.3f} |")
    lines.extend(["","## Repetitions and provenance",""])
    for label,report in zip(["Default","One-thread"],reports):
        lines.append(f"{label}: "+", ".join(f"run {x['run']}: {x['wall_fps']:.2f} FPS ({x['frames']} frames, {x['wall_seconds']:.2f} s)" for x in report['runs']))
    lines.extend(["",f"Software: {e['python']}; {e['packages']}. Platform: {e['platform']}. OpenCV threads: {e['opencv_threads']}.",
        f"Video SHA-256: {v['sha256']}. Model SHA-256: {r['model']['sha256']}.",
        f"Model checkpoint training freeze field: {reports[-1]['model'].get('training_freeze','not recorded')}. Verify the exact checkpoint against the paper's six-frozen-module description; a filename is not proof of training provenance.",
        "","## Required qualifications before publication", "",
        "- The supplied clip is 478 x 850, not the paper dataset's stated 720 x 1280. Do not describe these timings as measured at the larger resolution.",
        "- This is a separate paper-specific adapter: SG runs on both tracked axes, including Kalman-filled gaps. The dashboard currently computes measured Y derivatives only. Neither the hand checker nor the hit/miss processor is executed in this benchmark.",
        "- Local service time is not capture-to-display latency. Camera acquisition queues, network transmission, SSE and browser paint require a separate instrumented experiment.",
        "- The pixel-to-metre factor is inherited (2650). Velocity/acceleration values are estimates; physical accuracy and calibration for this clip have not been validated.",
        "- Counts of detected/predicted frames describe availability, not detection accuracy. Numeric samples include a provenance column; predictions are not ground truth.",
        "- The causal fit starts after three samples and grows to seven. It uses past samples only. Startup buffering is separate from computational service time; no full-window future-frame delay is imposed.",
        "- Do not use throughput results to infer the comparative speed of the other YOLO models in Table I; only the supplied weight file was benchmarked.",
        "- Keep the unmodified source PDF. Insert the supplied LaTeX subsection/table into the manuscript source; the PDF was not rewritten or reflowed.",
        "","## Reproduce","","```powershell","py -3 scripts/benchmark_paper.py","py -3 scripts/benchmark_paper.py --threads 1 --output outputs/paper_benchmark_threads1","py -3 scripts/build_paper_report.py","```",""])
    (out/"results_for_paper.md").write_text("\n".join(lines),encoding="utf-8")
    tex="\\subsection{Computational Performance and Latency}\n"+paragraph.replace("%",r"\%")+"\n\n"
    tex+=r"\begin{table}[t]"+"\n"+r"\centering"+"\n"+r"\caption{CPU performance through velocity and acceleration estimation. Local timings include decoding, annotation and JPEG encoding, but exclude network and browser delays.}"+"\n"+r"\label{tab:runtime}"+"\n"+r"\begin{tabular}{lrrrr}"+"\n"+r"\hline"+"\n"+r"Setup & Threads & Core FPS & Local FPS & P95 (ms) \\"+"\n"+r"\hline"+"\n"
    for row in table:tex+=" & ".join([row[0],row[1],row[2],row[3],row[5]])+r" \\"+"\n"
    tex+=r"\hline"+"\n"+r"\end{tabular}"+"\n"+r"\end{table}"+"\n"
    (out/"performance_section.tex").write_text(tex,encoding="utf-8")
    frames=list(csv.DictReader((folders[best_index]/"frame_timings.csv").open()))
    first=[row for row in frames if row['run']=='1']
    times=np.array([float(row['local_frame_ms']) for row in first])
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot(np.arange(1,len(times)+1),times,lw=.8)
    axes[0].axhline(r['source_deadline_ms'],color='red',ls='--',label='Source frame budget')
    axes[0].set(xlabel='Frame (pass 1)',ylabel='Local service time (ms)');axes[0].legend()
    means=[s[k]['mean_ms'] for k in ['decode_ms','detection_ms','tracking_ms','motion_ms','drawing_ms','jpeg_ms']]
    axes[1].bar(['Decode','YOLO','Kalman','SG + motion','Draw','JPEG'],means)
    axes[1].set(ylabel='Mean stage time (ms)');axes[1].tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(out/'runtime_results.png',dpi=180);plt.close(fig)
    motion=list(csv.DictReader((folders[best_index]/'motion_samples.csv').open()))
    valid=[x for x in motion if x['vy_m_s']]
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
    for field,label in [('vx_m_s','Vx'),('vy_m_s','Vy')]:axes[0].plot([float(x['video_time_s']) for x in valid],[float(x[field]) for x in valid],label=label,lw=.8)
    for field,label in [('ax_m_s2','Ax'),('ay_m_s2','Ay')]:axes[1].plot([float(x['video_time_s']) for x in valid],[float(x[field]) for x in valid],label=label,lw=.8)
    axes[0].set(ylabel='Estimated velocity (m/s)');axes[1].set(ylabel='Estimated acceleration (m/s2)',xlabel='Video time (s)')
    for ax in axes:
        flagged=False
        for row in valid:
            if row['source']=='predicted':
                time=float(row['video_time_s'])
                ax.axvspan(time-0.5/v['source_fps'],time+0.5/v['source_fps'],color='gray',alpha=.15,
                           label='Kalman-predicted sample' if not flagged else None)
                flagged=True
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'motion_estimates.png',dpi=180);plt.close(fig)
    print('Report:',out)
    print(verdict)
    print(markdown)

if __name__=='__main__':main()
