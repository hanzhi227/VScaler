import Foundation
import MacVideoUpscalerCore

// Headless runner: decode -> 2x Metal upscale -> VideoToolbox -> MPEG-TS over HTTP.
// Status lines (JSON) on stdout. Example:
//   macvideoupscale --file ep.mkv --mode cnn --port 8090 --open-vlc

let args = ProcessInfo.processInfo.arguments
func arg(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func flag(_ name: String) -> Bool { args.contains(name) }

guard let filePath = arg("--file") else {
    FileHandle.standardError.write(Data("""
    usage: macvideoupscale --file <video> [--mode cnn|cnnDenoise|bypass|metalfx|realesrgan|off] [--scale 2|4] [--port 8090]
                        [--start 0] [--ring-mb 128] [--prebuffer-mb 4]
                        [--no-audio] [--audio-index N]
                        [--subtitles] [--sub-index N] [--open-vlc]
    """.utf8))
    exit(2)
}

var cfg = PipelineController.Config()
if let v = arg("--port"), let p = UInt16(v) { cfg.port = p }
if let v = arg("--ring-mb"), let m = Int(v) { cfg.ringCapacityBytes = m << 20 }
if let v = arg("--prebuffer-mb"), let m = Int(v) { cfg.prebufferBytes = m << 20 }
if let v = arg("--lead-seconds"), let s = Double(v) { cfg.leadSeconds = max(5, s) }
if flag("--no-audio") { cfg.audioEnabled = false }
if let v = arg("--audio-index"), let i = Int(v) { cfg.audioStreamIndex = i }
if let v = arg("--scale"), let n = Int(v), let f = UpscaleFactor(rawValue: n) { cfg.scaleFactor = f }
if flag("--subtitles") { cfg.burnSubtitles = true }
if let v = arg("--sub-index"), let i = Int(v) { cfg.subtitleStreamIndex = i }

let kind = ScalerKind.parse(arg("--mode"))

guard let controller = try? PipelineController(config: cfg) else {
    FileHandle.standardError.write(Data("failed to init pipeline (Metal/server)\n".utf8))
    exit(1)
}

controller.status.on_change = { snap in
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    if let line = try? enc.encode(snap), var s = String(data: line, encoding: .utf8) {
        s.append("\n")
        FileHandle.standardOutput.write(s.data(using: .utf8)!)
    }
}

signal(SIGINT, SIG_IGN)
let sigSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
sigSrc.setEventHandler {
    FileHandle.standardError.write(Data("[cli] stopping…\n".utf8))
    controller.stop()
    exit(0)
}
sigSrc.resume()

// Also handle SIGTERM: without this, killing the engine skipped its cleanup and
// left the owned VLC running (holding its HTTP port and, worse, answering the
// next session's clock queries about a dead stream).
signal(SIGTERM, SIG_IGN)
let termSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termSrc.setEventHandler {
    controller.stop()
    exit(0)
}
termSrc.resume()

if flag("--open-vlc") {
    DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
        // The engine owns this VLC so it can read the playback clock and reopen
        // the stream itself after a seek.
        _ = controller.startPlayback()
    }
}

do {
    try controller.open(file: URL(fileURLWithPath: filePath), kind: kind,
                        startAt: Double(arg("--start") ?? "0") ?? 0)
} catch {
    FileHandle.standardError.write(Data("open failed: \(error)\n".utf8))
    exit(1)
}

// Headless control scripting for validation, e.g.
//   --script "5:pause,8:resume,12:seek:60,20:mode:bypass"
// Emits {"event":...} JSON lines so control effects are verifiable.
var controlTimer: DispatchSourceTimer?
if let script = arg("--script") {
    struct Action { let at: Double; let verb: String; let value: String?; var fired = false }
    var actions: [Action] = script.split(separator: ",").compactMap { item in
        let parts = item.split(separator: ":")
        guard let at = Double(parts[0]) else { return nil }
        return Action(at: at, verb: String(parts[1]),
                      value: parts.count > 2 ? String(parts[2]) : nil)
    }
    let start = Date()
    let timer = DispatchSource.makeTimerSource(queue: .main)
    timer.schedule(deadline: .now(), repeating: 0.2)
    timer.setEventHandler {
        let elapsed = Date().timeIntervalSince(start)
        for i in actions.indices where !actions[i].fired && elapsed >= actions[i].at {
            actions[i].fired = true
            let a = actions[i]
            // Capture what the control was based on BEFORE it runs: a relative
            // seek must be anchored to the playback estimate, not the production
            // frontier, and this is what makes that checkable.
            let before = controller.status.current
            switch a.verb {
            case "pause": controller.pause()
            case "resume": controller.resume()
            case "seek": controller.seek(to: Double(a.value ?? "0") ?? 0)
            case "seekrel": controller.seek(by: Double(a.value ?? "0") ?? 0)
            case "play": _ = controller.startPlayback()
            case "audio": controller.setAudioTrack(Int(a.value ?? "0") ?? 0)
            case "scale":
                if let n = Int(a.value ?? "2"), let f = UpscaleFactor(rawValue: n) {
                    controller.setScaleFactor(f)
                }
            case "subtitles": controller.setSubtitles(!controller.config.burnSubtitles)
            case "subs-off": controller.setSubtitles(false)
            case "subs-on": controller.setSubtitleTrack(Int(a.value ?? "0") ?? 0)
            case "mode": controller.setKind(ScalerKind.parse(a.value))
            default: break
            }
            let snap = controller.status.current
            let ev: [String: Any] = [
                "event": a.verb, "at": a.at, "wall": Date().timeIntervalSince(start),
                "value": a.value ?? "",
                "state": snap.state.rawValue, "session": snap.sessionId,
                "position": snap.positionSeconds, "frames": snap.framesProcessed,
                "mode": snap.mode, "stream": controller.streamURL.absoluteString,
                "playback_before": before.playbackSeconds,
                "frontier_before": before.positionSeconds,
                "target": snap.positionSeconds,
            ]
            if let d = try? JSONSerialization.data(withJSONObject: ev, options: [.sortedKeys]) {
                FileHandle.standardOutput.write(d + Data("\n".utf8))
            }
        }
    }
    timer.resume()
    controlTimer = timer  // must be retained: a released dispatch source is cancelled
}

// Park the main thread; status callbacks do the printing.
dispatchMain()
