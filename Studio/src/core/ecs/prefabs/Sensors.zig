const std = @import("std");
const Math = @import("../../Math.zig");

/// Image sensor physical specifications
pub const ImageSensor = struct {
    name: []const u8,
    sensor_width_mm: f32,
    sensor_height_mm: f32,
    pixel_pitch_um: f32,
    native_width: u32,
    native_height: u32,
};

/// Camera module configuration (sensor + lens combination)
pub const CameraModule = struct {
    name: []const u8,
    sensor: ImageSensor,
    focal_length_mm: f32,
    hfov_degrees: f32, // Horizontal FOV

    /// Compute vertical FOV from sensor aspect and horizontal FOV
    pub fn vfov(self: CameraModule) f32 {
        const aspect = self.sensor.sensor_width_mm / self.sensor.sensor_height_mm;
        const hfov_rad = Math.radians(self.hfov_degrees);
        const vfov_rad = 2.0 * std.math.atan(@tan(hfov_rad / 2.0) / aspect);
        return Math.degrees(vfov_rad);
    }

    /// Compute fx (horizontal focal length in pixels) for given resolution
    pub fn fx(self: CameraModule, resolution_width: u32) f32 {
        const pixels_per_mm = @as(f32, @floatFromInt(resolution_width)) / self.sensor.sensor_width_mm;
        return self.focal_length_mm * pixels_per_mm;
    }

    /// Compute fy (vertical focal length in pixels) for given resolution
    pub fn fy(self: CameraModule, resolution_height: u32) f32 {
        const pixels_per_mm = @as(f32, @floatFromInt(resolution_height)) / self.sensor.sensor_height_mm;
        return self.focal_length_mm * pixels_per_mm;
    }

    /// Check if pixels are square (within 1% tolerance)
    pub fn hasSquarePixels(self: CameraModule) bool {
        const width_pitch = self.sensor.sensor_width_mm * 1000.0 / @as(f32, @floatFromInt(self.sensor.native_width));
        const height_pitch = self.sensor.sensor_height_mm * 1000.0 / @as(f32, @floatFromInt(self.sensor.native_height));
        const ratio = width_pitch / height_pitch;
        return ratio > 0.99 and ratio < 1.01;
    }
};

// ============================================================================
// Sensor Definitions
// ============================================================================

pub const Sensor = struct {
    /// Sony IMX708 - 12.3MP, 1.4µm pixels, 1/2.43" format
    pub const IMX708 = ImageSensor{
        .name = "Sony IMX708",
        .sensor_width_mm = 6.45,
        .sensor_height_mm = 4.84,
        .pixel_pitch_um = 1.4,
        .native_width = 4608,
        .native_height = 3456,
    };

    /// Sony IMX219 - 8MP, 1.12µm pixels, 1/4" format
    pub const IMX219 = ImageSensor{
        .name = "Sony IMX219",
        .sensor_width_mm = 3.68,
        .sensor_height_mm = 2.76,
        .pixel_pitch_um = 1.12,
        .native_width = 3280,
        .native_height = 2464,
    };

    /// OmniVision OV5647 - 5MP, 1.4µm pixels
    pub const OV5647 = ImageSensor{
        .name = "OmniVision OV5647",
        .sensor_width_mm = 3.76,
        .sensor_height_mm = 2.74,
        .pixel_pitch_um = 1.4,
        .native_width = 2592,
        .native_height = 1944,
    };
};

// ============================================================================
// Camera Module Presets (Sensor + Lens combinations)
// ============================================================================

pub const Module = struct {
    /// Raspberry Pi Camera Module 3 - Standard (66° HFOV)
    pub const PiCam3_Standard = CameraModule{
        .name = "Pi Camera Module 3 Standard",
        .sensor = Sensor.IMX708,
        .focal_length_mm = 4.74,
        .hfov_degrees = 66.0,
    };

    /// Raspberry Pi Camera Module 3 - Wide (102° HFOV)
    pub const PiCam3_Wide = CameraModule{
        .name = "Pi Camera Module 3 Wide",
        .sensor = Sensor.IMX708,
        .focal_length_mm = 2.75,
        .hfov_degrees = 102.0,
    };

    /// Raspberry Pi Camera Module 3 - Ultra Wide (120° HFOV)
    pub const PiCam3_UltraWide = CameraModule{
        .name = "Pi Camera Module 3 Ultra Wide",
        .sensor = Sensor.IMX708,
        .focal_length_mm = 1.95,
        .hfov_degrees = 120.0,
    };

    /// Raspberry Pi Camera Module 2 - Standard (62° HFOV)
    pub const PiCam2_Standard = CameraModule{
        .name = "Pi Camera Module 2",
        .sensor = Sensor.IMX219,
        .focal_length_mm = 3.04,
        .hfov_degrees = 62.2,
    };

    /// Raspberry Pi Camera Module 1 (OV5647)
    pub const PiCam1 = CameraModule{
        .name = "Pi Camera Module 1",
        .sensor = Sensor.OV5647,
        .focal_length_mm = 3.6,
        .hfov_degrees = 53.5,
    };
};

/// Default camera module
pub const Default = Module.PiCam3_Wide;
