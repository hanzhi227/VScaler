-- Screenshot helper for headless benchmarking with GPU VOs.
-- Usage: mpv --script=shot.lua --script-opts=shot-file=/abs/out.png,shot-mode=video ...
-- Takes one screenshot of the rendered (post-glsl-shader) frame, then quits.
local file = mp.get_opt("shot-file")
local mode = mp.get_opt("shot-mode") or "video"
local pause_s = tonumber(mp.get_opt("shot-delay") or "0.25")
local done = false

local function take()
    if done then return end
    done = true
    if file then
        mp.commandv("screenshot-to-file", file, mode)
    end
    mp.commandv("quit")
end

local function schedule()
    mp.add_timeout(pause_s, take)
end

mp.register_event("playback-restart", schedule)
mp.register_event("file-loaded", schedule)
-- Safety net: never hang forever.
mp.add_timeout(15, take)
