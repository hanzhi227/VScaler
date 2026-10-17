// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "MacVideoUpscaler",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MacVideoUpscalerCore", targets: ["MacVideoUpscalerCore"]),
        .executable(name: "macvideoupscale", targets: ["macvideoupscale"]),
        .executable(name: "MacVideoUpscalerApp", targets: ["MacVideoUpscalerApp"]),
        .executable(name: "playercheck", targets: ["playercheck"]),
    ],
    dependencies: [
        .package(path: "../VideoUpscaleMetal"),
        .package(path: "../RealESRGANMetal"),
    ],
    targets: [
        .target(
            name: "MacVideoUpscalerCore",
            dependencies: [
                .product(name: "VideoUpscaleMetal", package: "VideoUpscaleMetal"),
                .product(name: "RealESRGANMetal", package: "RealESRGANMetal"),
            ],
            swiftSettings: [.define("HAVE_VIDEO_UPSCALE_METAL")]
        ),
        .testTarget(
            name: "MacVideoUpscalerCoreTests",
            dependencies: ["MacVideoUpscalerCore"]
        ),
        .executableTarget(
            name: "macvideoupscale",
            dependencies: ["MacVideoUpscalerCore"]
        ),
        .executableTarget(
            name: "playercheck",
            dependencies: ["MacVideoUpscalerCore"]
        ),
        .executableTarget(
            name: "MacVideoUpscalerApp",
            dependencies: ["MacVideoUpscalerCore"]
        ),
    ]
)
