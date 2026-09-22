-- Print end-of-playback video stats as a single parseable line.
local props = {
    "vo-drop-frame-count", "mistimed-frame-count", "frame-drop-count",
    "decoder-frame-drop-count", "vo-delayed-frame-count",
    "estimated-vf-fps", "display-fps", "estimated-frame-number",
    "video-speed-correction",
}
local function dump()
    local parts = {}
    for _, k in ipairs(props) do
        local v = mp.get_property_native(k)
        parts[#parts+1] = k .. "=" .. tostring(v)
    end
    print("STATS " .. table.concat(parts, " "))
end
mp.register_event("end-file", dump)
mp.register_event("shutdown", dump)
