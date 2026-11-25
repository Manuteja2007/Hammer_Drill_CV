"use strict";

const App = (() => {
    const $ = (id) => document.getElementById(id);
    const fields = ["v_y", "accel_y", "y_smooth", "state", "direction", "confidence", "ball_x", "ball_y",
        "max_v_y", "avg_v_y", "hit_count", "miss_count", "current_streak", "best_streak",
        "hand_status", "hand_check_ms", "rejected_count"];
    let source = null;
    let currentMode = null;
    let active = false;
    let videoReady = false;
    let duration = 0;
    let playState = "pause";
    let scrubbing = false;
    let toastTimer = null;
    let pending = false;
    let sessionLabel = "Hammer drill session";
    let tapGoal = 0;
    let lastSummary = null;
    let reviewCancelled = false;

    function goalUpdate(mode, count) {
        $(mode + "-goal-label").textContent = tapGoal ? `${count} of ${tapGoal} taps${count >= tapGoal ? " - target reached" : ""}` : "No tap target set";
        $(mode + "-goal-progress").value = tapGoal ? Math.min(100, 100 * count / tapGoal) : 0;
    }

    function metric(key, value) {
        if (["v_y", "accel_y", "y_smooth", "ball_x", "ball_y", "max_v_y", "avg_v_y"].includes(key)) {
            return value !== "--" && value !== null && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "--";
        }
        if (key === "confidence" && value !== "--" && Number.isFinite(Number(value))) return `${(Number(value) * 100).toFixed(0)}%`;
        return value;
    }

    function playbackStatus(mode, data) {
        const labels = { "Tracking": "Ball tracked", "Lost/Predicting": "Ball out of view", "Buffering": "Finding the ball", "Video Ended": "Review complete", "Camera Disconnected": "Camera disconnected", "Processing Error": "Analysis interrupted" };
        $(mode + "-state").textContent = labels[data.state] || data.state || "Waiting";
        $(mode + "-playback-label").textContent = data.state === "Video Ended" ? "Recording ended" : data.play_state === "play" ? "Analyzing" : "Paused";
        const cues = { "Lost/Predicting": "Ball tracking is interrupted. Keep the ball inside the frame and check the lighting.", "Buffering": "Finding the ball. Keep the ball and hammer head clearly visible.", "Video Ended": "The recording has ended. Finish review to see the summary, or select Play to analyze it again.", "Camera Disconnected": "The camera connection was lost. Try Resume, or finish this session.", "Processing Error": "Analysis stopped unexpectedly. Finish this session and try again." };
        $(mode + "-session-cue").textContent = cues[data.state] || (data.play_state === "pause" ? "Paused. Review the last frame or resume when ready." : "Focus on a steady tap-and-return rhythm. The counts are estimates; review flagged taps after the set.");
    }

    function toast(message) {
        clearTimeout(toastTimer);
        $("toast").textContent = message;
        $("toast").classList.remove("hidden");
        toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 4500);
    }

    async function api(url, options = {}) {
        const response = await fetch(url, options);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
        return data;
    }

    function control(mode, action, time = 0) {
        return api("/video_control", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode, action, time }) });
    }

    function formatTime(value, short = false) {
        const seconds = Math.max(0, Math.floor(Number(value) || 0));
        const minutes = Math.floor(seconds / 60);
        const tail = `${String(minutes % 60).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
        return short ? `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}` :
            `${String(Math.floor(seconds / 3600)).padStart(2, "0")}:${tail}`;
    }

    function view(id) {
        document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === id));
        const section = $(id);
        section.tabIndex = -1;
        if (typeof section.focus === "function") section.focus({ preventScroll: true });
    }

    function disconnect() {
        if (source) source.close();
        source = null;
        for (const mode of ["live", "video"]) $(mode + "-feed").removeAttribute("src");
    }

    function resetMetrics(mode) {
        for (const field of fields) $(mode + "-" + field).textContent =
            ["hit_count", "miss_count", "current_streak", "best_streak", "rejected_count", "hand_check_ms"].includes(field) ? "0" : "--";
        $(mode + "-hand_status").textContent = "Checks run at candidate taps";
        $(mode + "-state").textContent = "Waiting";
        $(mode + "-timeline_text").textContent = "No events yet";
        const badge = $(mode + "-last_event_badge");
        badge.textContent = "--";
        badge.className = "hm-badge";
        $(mode + "-timer").textContent = "00:00:00";
    }

    function connect(mode) {
        if (source) source.close();
        source = new EventSource("/stream_stats/" + mode);
        let warned = false;
        source.onerror = () => {
            if (!warned) toast("Analytics connection interrupted. Reconnecting...");
            warned = true;
        };
        source.onmessage = event => {
            warned = false;
            const data = JSON.parse(event.data);
            for (const key of fields) {
                if (data[key] !== undefined) $(mode + "-" + key).textContent = metric(key, data[key]);
            }
            $(mode + "-timeline_text").textContent = (data.timeline_text || "No events yet").replace(/hits/g, "taps").replace(/hit/g, "tap").replace(/MISS/g, "break").replace(/->/g, " / ");
            goalUpdate(mode, Number(data.hit_count) || 0);
            playbackStatus(mode, data);
            const badge = $(mode + "-last_event_badge");
            badge.textContent = data.last_event;
            badge.className = "hm-badge" + (data.last_event === "HIT" ? " hm-badge-hit" :
                data.last_event === "MISS" ? " hm-badge-miss" : data.last_event === "REJECTED" ? " hm-badge-rejected" : "");
            playState = data.play_state;
            
            $(mode === "live" ? "btn-live-pause" : "btn-video-play").textContent =
                playState === "play" ? "Pause" : mode === "live" ? "Resume" : "Play";
            $(mode + "-timer").textContent = formatTime(data.time);
            if (mode === "video") {
                duration = Number(data.duration) || 0;
                $("video-timer-total").textContent = formatTime(duration, true);
                if (!scrubbing) {
                    $("video-timer-current").textContent = formatTime(data.time, true);
                    $("video-scrubber").value = duration ? 100 * data.time / duration : 0;
                }
            }
        };
    }

    async function startFeed(mode) {
        resetMetrics(mode);
        $(mode + "-playback-label").textContent = "Starting analysis...";
        $(mode + "-session-cue").textContent = "Preparing the video analysis. The first session may take longer while the models load.";
        const data = await api("/start_stream/" + mode, { method: "POST" });
        active = true;
        playState = "play";
        const feed = $(mode + "-feed");
        feed.classList.remove("paused");
        feed.src = data.feed_url;
        if (mode === "live") $("live-placeholder").style.display = "none";
        else {
            $("video-scrubber").disabled = false;
            $("btn-video-play").disabled = false;
            $("btn-video-finish").disabled = false;
        }
        connect(mode);
    }

    async function goHome() {
        await Promise.all([control("live", "stop"), control("video", "stop")]);
        disconnect();
        active = videoReady = scrubbing = false;
        duration = 0;
        currentMode = null;
        lastSummary = null;
        $("session-notes").value = "";
        $("video-upload-input").value = "";
        $("upload-zone").style.display = "flex";
        $("upload-status").textContent = "Formats: MP4, AVI, MOV, MKV, WEBM (up to 500 MB)";
        view("view-landing");
    }

    function start(mode) {
        sessionLabel = $("session-label").value.trim() || "Hammer drill session";
        const goalValue = Number($("goal-taps").value);
        if (String($("goal-taps").value).trim() && (!Number.isInteger(goalValue) || goalValue < 1 || goalValue > 10000)) {
            throw new Error("Enter a whole-number tap target from 1 to 10,000, or leave it blank.");
        }
        tapGoal = Number.isInteger(goalValue) && goalValue > 0 && goalValue <= 10000 ? goalValue : 0;
        currentMode = mode === "live" ? "live" : "video";
        resetMetrics(currentMode);
        $(currentMode + "-session-name").textContent = sessionLabel;
        goalUpdate(currentMode, 0);
        $(currentMode + "-playback-label").textContent = "Ready";
        if (currentMode === "live") {
            $("btn-live-pause").disabled = true;
            $("btn-live-session").textContent = "Start Session";
            $("btn-live-session").className = "btn btn-primary";
            $("live-placeholder").style.display = "flex";
        } else {
            $("video-timer-current").textContent = "00:00";
            $("video-timer-total").textContent = "00:00";
            $("video-scrubber").value = 0;
            $("video-scrubber").disabled = true;
            $("btn-video-play").disabled = true;
            $("btn-video-finish").disabled = true;
            $("btn-video-play").textContent = "Play";
            $("review-progress").hidden = true;
            $("upload-zone").style.display = "flex";
        }
        view("view-" + currentMode);
    }

    async function toggleLiveSession() {
        if (active) return stopSession("live");
        $("btn-live-session").textContent = "Starting...";
        try { await startFeed("live"); }
        catch (error) {
            $("btn-live-session").textContent = "Start Session";
            $("live-session-cue").textContent = error.message;
            $("live-playback-label").textContent = "Could not start";
            throw error;
        }
        $("btn-live-pause").disabled = false;
        $("btn-live-session").textContent = "Finish session";
        $("btn-live-session").className = "btn btn-danger";
    }

    async function togglePlayback(mode) {
        if (!active) {
            if (mode === "video" && videoReady) return startFeed(mode);
            throw new Error(mode === "video" ? "Upload a video first." : "Start a session first.");
        }
        const nextState = playState === "play" ? "pause" : "play";
        await control(mode, nextState);
        playState = nextState;
    }

    async function handleUpload() {
        const input = $("video-upload-input");
        if (!input.files.length) return;
        const file = input.files[0];
        if (file.size > 500 * 1024 * 1024) throw new Error("Video exceeds the 500 MB upload limit.");
        input.disabled = true;
        $("upload-status").textContent = `Uploading ${file.name}...`;
        try {
            const form = new FormData();
            form.append("video", file);
            await api("/upload_video", { method: "POST", body: form });
            videoReady = true;
            duration = 0;
            $("upload-zone").style.display = "none";
            await startFeed("video");
            toast("Video uploaded. Analysis started.");
        } catch (error) {
            $("upload-status").textContent = error.message;
            $("upload-zone").style.display = "flex";
            $("video-session-cue").textContent = error.message;
            throw error;
        } finally {
            input.disabled = false;
        }
    }

    async function seekVideo(percent) {
        scrubbing = false;
        if (!videoReady || !active || duration <= 0) throw new Error("Start a video before seeking.");
        await control("video", "seek", Number(percent) * duration / 100);
        await control("video", "play");
        playState = "play";
        toast("Preview moved. Finish review will analyze the entire recording.");
    }

    function buildTimeline(timeline) {
        const container = $("res-timeline-visual");
        container.replaceChildren();
        const segments = [];
        for (const event of timeline) {
            const previous = segments[segments.length - 1];
            if (event.type === "HIT" && previous?.type === "HIT") previous.count++;
            else segments.push({ type: event.type, count: 1, reason: event.reason });
        }
        if (!segments.length) container.textContent = "No events recorded";
        segments.forEach((segment, index) => {
            if (index) {
                const arrow = document.createElement("span");
                arrow.className = "tl-arrow";
                arrow.textContent = " / ";
                container.append(arrow);
            }
            const element = document.createElement("span");
            element.className = "tl-segment " + (segment.type === "HIT" ? "tl-hit-seg" : "tl-miss-seg");
            element.textContent = segment.type === "HIT" ?
                `${segment.count} hit${segment.count === 1 ? "" : "s"}` : "MISS";
            element.title = segment.reason || "";
            container.append(element);
        });
    }

    async function stopSession(mode) {
        if (!active && !(mode === "video" && videoReady)) throw new Error("Start an analysis session first.");
        await control(mode, "stop");
        let data;
        if (mode === "video") {
            disconnect();
            active = false;
            reviewCancelled = false;
            $("review-progress").hidden = false;
            $("btn-video-finish").disabled = true;
            $("btn-video-play").disabled = true;
            $("video-scrubber").disabled = true;
            try {
                let job = await api("/review_video", { method: "POST" });
                while (job.status === "running" && !reviewCancelled) {
                    const processed = Number(job.processed_frames) || 0;
                    const total = Number(job.total_frames) || 0;
                    $("review-progress-label").textContent = `Analyzing the entire recording: ${processed}${total ? ` / ${total}` : ""} frames`;
                    $("review-progress-bar").value = total ? Math.min(100, 100 * processed / total) : 0;
                    await new Promise(resolve => setTimeout(resolve, 500));
                    job = await api("/review_video");
                }
                if (reviewCancelled || job.status === "cancelled") {
                    await api("/cancel_review", { method: "POST" });
                    return;
                }
                if (job.status !== "complete") throw new Error(job.error || "Full recording analysis failed. Please retry.");
                data = job.result;
            } finally {
                $("review-progress").hidden = true;
                $("btn-video-finish").disabled = false;
                $("btn-video-play").disabled = false;
            }
        } else data = await api("/session_summary/" + mode);
        lastSummary = data;
        $("results-session-name").textContent = sessionLabel;
        const taps = Number(data.total_hits) || 0;
        const best = Number(data.best_streak) || 0;
        $("results-session-insight").textContent = taps ? `${best} taps in the longest streak, from ${taps} estimated taps.${tapGoal ? ` Target: ${tapGoal} taps${taps >= tapGoal ? " reached" : " not reached"}.` : ""} Review any hand-contact flags before using the count as a performance measure.` : "No taps were counted. Review the camera angle, ball visibility and hand-check results before repeating the drill.";
        disconnect();
        active = false;
        $("res-duration").textContent = formatTime(data.elapsed);
        $("res-rejected-count").textContent = data.rejected_count || 0;
        $("res-hand-contact-count").textContent = data.hand_rejected_count || 0;
        $("res-unverified-count").textContent = data.unverified_count || 0;
        $("res-uncertain-hit-count").textContent = data.uncertain_hit_count || 0;
        for (const [id, key] of Object.entries({ "res-max-v": "max_velocity", "res-avg-v": "avg_velocity",
            "res-avg-a": "avg_accel", "res-total-hits": "total_hits", "res-total-misses": "total_misses",
            "res-best-streak": "best_streak", "res-timeline-text": "timeline_text" })) {
            $(id).textContent = data[key];
        }
        buildTimeline(data.timeline);
        view("view-results");
    }

    async function cancelReview() {
        try {
            reviewCancelled = true;
            await api("/cancel_review", { method: "POST" });
            toast("Full recording analysis cancelled. You can retry Finish review.");
        } catch (error) { toast(error.message); }
    }

    async function downloadReport() {
        if (!lastSummary) throw new Error("Finish a session before downloading its report.");
        const report = { session: sessionLabel, tap_target: tapGoal || null, ...lastSummary,
            session_notes: $("session-notes").value.slice(0, 4000),
            interpretation: "Tap counts and hand contact are estimates. Streak breaks are inferred, not verified missed strikes. Motion units use a preset camera scale." };
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = "hammer-drill-session.json";
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    const guard = fn => async (...args) => {
        if (pending) return;
        pending = true;
        try { return await fn(...args); }
        catch (error) { toast(error.message); }
        finally { pending = false; }
    };
    document.addEventListener("DOMContentLoaded", () => {
        $("video-scrubber").addEventListener("input", event => {
            scrubbing = true;
            $("video-timer-current").textContent = formatTime(Number(event.target.value) * duration / 100, true);
        });
        $("video-scrubber").addEventListener("change", guard(event => seekVideo(event.target.value)));
        for (const mode of ["live", "video"]) {
            $(mode + "-feed").addEventListener("error", guard(async () => {
                if (!active || currentMode !== mode) return;
                await control(mode, "stop");
                disconnect();
                active = false;
                if (mode === "live") {
                    $("btn-live-pause").disabled = true;
                    $("btn-live-session").textContent = "Start Session";
                }
                toast("Unable to open the stream. Check the model, video, or camera availability, then retry.");
            }));
        }
    });
    window.addEventListener("pagehide", () => {
        navigator.sendBeacon("/cancel_review");
        if (active && currentMode) navigator.sendBeacon("/video_control", new Blob([
            JSON.stringify({ mode: currentMode, action: "stop" })], { type: "application/json" }));
    });
    window.addEventListener("pageshow", event => { if (event.persisted) guard(goHome)(); });
    return { start: guard(start), goHome: guard(goHome), toggleLiveSession: guard(toggleLiveSession),
        toggleLivePause: guard(() => togglePlayback("live")), handleUpload: guard(handleUpload),
        toggleVideoState: guard(() => togglePlayback("video")), seekVideo: guard(seekVideo),
        cancelReview, stopSession: guard(stopSession), downloadReport: guard(downloadReport) };
})();
