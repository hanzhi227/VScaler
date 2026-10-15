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

// A numeric flag that is present but unparseable is a usage error rather than
// something to ignore: running with a different value than the one asked for is
// harder to diagnose than refusing to start.
func intFlag(_ name: String) -> Int? {
    guard let v = arg(name) else { return nil }
    guard let n = Int(v), n >= 0 else {
        FileHandle.standardError.write(Data("\(name) expects a non-negative integer, got \"\(v)\"\n".utf8))
        exit(2)
    }
    return n
}
func doubleFlag(_ name: String) -> Double? {
    guard let v = arg(name) else { return nil }
    guard let n = Double(v), n.isFinite else {
        FileHandle.standardError.write(Data("\(name) expects a finite number, got \"\(v)\"\n".utf8))
        exit(2)
    }
    return n
}

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
if let p = intFlag("--port") {
    guard (1...65535).contains(p) else {
        FileHandle.standardError.write(Data("--port must be between 1 and 65535, got \(p)\n".utf8))
        exit(2)
    }
    cfg.port = UInt16(p)
}
// The shift is bounded so an absurd value errors out instead of trapping on
// overflow like the script items below used to.
if let m = intFlag("--ring-mb") {
    guard m > 0, m <= Int.max >> 20 else {
        FileHandle.standardError.write(Data("--ring-mb must be between 1 and \(Int.max >> 20)\n".utf8))
        exit(2)
    }
    cfg.ringCapacityBytes = m << 20
}
if let m = intFlag("--prebuffer-mb") {
    guard m <= Int.max >> 20 else {
        FileHandle.standardError.write(Data("--prebuffer-mb must not exceed \(Int.max >> 20)\n".utf8))
        exit(2)
    }
    cfg.prebufferBytes = m << 20
}
if let s = doubleFlag("--lead-seconds") { cfg.leadSeconds = max(5, s) }
if flag("--no-audio") { cfg.audioEnabled = false }
if let i = intFlag("--audio-index") { cfg.audioStreamIndex = i }
if let v = arg("--scale") {
    guard let n = Int(v), let f = UpscaleFactor(rawValue: n) else {
        FileHandle.standardError.write(Data("--scale must be 2 or 4, got \"\(v)\"\n".utf8))
        exit(2)
    }
    cfg.scaleFactor = f
}
if flag("--subtitles") { cfg.burnSubtitles = true }
if let i = intFlag("--sub-index") { cfg.subtitleStreamIndex = i }

let kind = ScalerKind.parse(arg("--mode"))

guard let controller = try? PipelineController(config: cfg) else {
    FileHandle.standardError.write(Data("failed to init pipeline (Metal/server)\n".utf8))
    exit(1)
}

// The store may post .failed more than once (every update reaches the handler),
// so the stop-and-exit below must fire on the first one only.
let failedExitGate = NSLock()
var failedExitTaken = false

controller.status.on_change = { snap in
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    if let line = try? enc.encode(snap), var s = String(data: line, encoding: .utf8) {
        s.append("\n")
        FileHandle.standardOutput.write(s.data(using: .utf8)!)
    }
    // A failed open (or a failure mid-stream) is reported as a status, not a
    // thrown error, so the process has to leave here: parking in dispatchMain()
    // below would hang a supervisor waiting on this exit code.
    //
    // This handler runs synchronously on the thread that called status.update,
    // which holds `ctrl` for every failure posted from beginSessionLocked.
    // controller.stop() locks that same non-recursive condition, so stopping
    // here would self-deadlock and the process could no longer even be killed.
    // Hop off the engine thread for the stop and the exit, gated so a repeated
    // .failed post cannot stop twice.
    if snap.state == .failed {
        FileHandle.standardError.write(
            Data("[cli] failed: \(snap.message.isEmpty ? "unknown error" : snap.message)\n".utf8))
        failedExitGate.lock()
        let first = !failedExitTaken
        failedExitTaken = true
        failedExitGate.unlock()
        guard first else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            controller.stop()
            exit(1)
        }
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
        // the stream itself after a seek. Skipped when the open already failed:
        // there is no stream for VLC to play.
        guard controller.status.current.state != .failed else { return }
        _ = controller.startPlayback()
    }
}

// open cannot throw: a file that fails to decode reports it through the
// .failed status handled in on_change above.
try? controller.open(file: URL(fileURLWithPath: filePath), kind: kind,
                     startAt: doubleFlag("--start") ?? 0)

// Headless control scripting for validation, e.g.
//   --script "5:pause,8:resume,12:seek:60,20:mode:bypass"
// Emits {"event":...} JSON lines so control effects are verifiable.
var controlTimer: DispatchSourceTimer?
if let script = arg("--script") {
    struct Action { let at: Double; let verb: String; let value: String?; var fired = false }
    var actions: [Action] = script.split(separator: ",").compactMap { item in
        let parts = item.split(separator: ":")
        // Count checked before the subscript: an item with no colon (or only
        // colons) must be skipped, not trap on an out-of-range index.
        guard parts.count >= 2, let at = Double(parts[0]) else { return nil }
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
