# Manuscript wording to match the measured results

Insert performance_section.tex as Section IV.D (after the motion-analytics results), and include runtime_results.png if space allows. Reflow and compile from the original LaTeX source; main.pdf is preserved.

The supplied clip's 29.39 FPS requires about 34.02 ms/frame. The best local service time was 63.41 ms/frame, with 15.77 FPS wall throughput. Do not retain an unqualified claim of real-time processing at the input frame rate on this CPU.

Suggested abstract replacement:

> Experimental evaluation demonstrates automated detection, tracking, and motion estimation on recorded indoor drill videos. On the tested Intel Core i5-1135G7 CPU, the complete local processing path achieved 15.77 FPS, with a mean frame service time of 63.41 ms; sustaining the input frame rate requires further deployment optimization.

Suggested conclusion replacement:

> The framework provides a practical basis for automated hammer-drill motion analysis. The measured CPU implementation supports recorded-video processing, while full-frame-rate real-time deployment remains subject to further optimization and hardware evaluation.

Additional checks before submission:

- The timed clip is 478 x 850. The paper's 720 x 1280 describes a different dataset setting and must not be substituted for the benchmark resolution.
- The supplied checkpoint records train_args.freeze = 10. The paper describes six frozen backbone modules. Verify the exact trained checkpoint and freezing convention before identifying these timings as those of the paper's selected YOLOv8n-LF model.
- The experiment times X/Y motion estimation in a separate paper-specific adapter, including Kalman-filled gaps. It does not time the current dashboard's measured-Y-only branch.
- Mean frame service time is not camera-to-screen latency. Do not claim the browser/network/camera stages were benchmarked.
- The calibration factor is inherited, not verified for this clip. The motion plot demonstrates estimates, not validated physical accuracy.
- No hand checking, impact counting, or related event logic is run by this benchmark. There is no need to introduce those future-work features into this results subsection.
