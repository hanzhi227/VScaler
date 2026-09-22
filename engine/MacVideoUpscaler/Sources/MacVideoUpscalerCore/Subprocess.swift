import Foundation

/// Reaping helper for the ffmpeg/ffprobe children.
enum Subprocess {
    /// Waits for `process` to exit **without** calling `Process.waitUntilExit()`.
    ///
    /// `waitUntilExit()` runs a nested run loop, so on the main thread it lets
    /// queued work — a menu action, a key press, a timer — run in the middle of a
    /// seek. A seek holds the controller's mutex while it waits, so that call
    /// re-enters and blocks on the mutex, and the run loop it is nested inside can
    /// never unwind: the app deadlocks, spinning, with the picture frozen.
    /// Polling cannot re-enter anything.
    ///
    /// `Process` reaps the child on its own and still fires `terminationHandler`,
    /// so nothing is lost by not calling `waitUntilExit()`.
    ///
    /// - Parameter timeout: how long to let the child exit gracefully before it is
    ///   killed. Callers pass 0 when they have already waited.
    static func wait(for process: Process, timeout: TimeInterval) {
        let start = Date()
        while process.isRunning {
            let waited = Date().timeIntervalSince(start)
            if waited >= timeout { break }
            // Tight at first — a seek waits here and a 20 ms poll is visible as
            // latency — then relaxed, because some callers wait out a whole
            // stream tail rather than a decoder teardown.
            usleep(waited < 0.1 ? 2_000 : 20_000)
        }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        // SIGKILL still has to be reaped before the object reports it stopped.
        let hardDeadline = Date().addingTimeInterval(1.0)
        while process.isRunning && Date() < hardDeadline { usleep(10_000) }
    }
}
