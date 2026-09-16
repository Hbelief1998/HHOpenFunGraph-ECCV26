from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageTransform2D:
    raw_height: int
    raw_width: int
    resized_height: int
    resized_width: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    downsample: int
    inference_height: int
    inference_width: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def scale_x(self) -> float:
        return float(self.resized_width) / float(self.raw_width)

    @property
    def scale_y(self) -> float:
        return float(self.resized_height) / float(self.raw_height)

    def raw_box_to_inference(self, box_xyxy) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        d = float(max(1, self.downsample))
        return [
            (x1 * self.scale_x - self.crop_left) / d,
            (y1 * self.scale_y - self.crop_top) / d,
            (x2 * self.scale_x - self.crop_left) / d,
            (y2 * self.scale_y - self.crop_top) / d,
        ]

    def inference_box_to_raw(self, box_xyxy) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        d = float(max(1, self.downsample))
        return [
            (x1 * d + self.crop_left) / self.scale_x,
            (y1 * d + self.crop_top) / self.scale_y,
            (x2 * d + self.crop_left) / self.scale_x,
            (y2 * d + self.crop_top) / self.scale_y,
        ]

    def raw_mask_to_inference(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.raw_height, self.raw_width):
            raise ValueError(f"raw mask shape must be {(self.raw_height, self.raw_width)}, got {mask.shape}")
        resized = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (self.resized_width, self.resized_height),
                Image.Resampling.NEAREST,
            )
        ) > 0
        crop = resized[
            self.crop_top : self.crop_top + self.crop_height,
            self.crop_left : self.crop_left + self.crop_width,
        ]
        d = max(1, int(self.downsample))
        return crop[::d, ::d].astype(bool)

    def inference_mask_to_raw(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.inference_height, self.inference_width):
            raise ValueError(
                f"inference mask shape must be {(self.inference_height, self.inference_width)}, got {mask.shape}"
            )
        d = max(1, int(self.downsample))
        if d > 1:
            crop_mask = np.zeros((self.crop_height, self.crop_width), dtype=bool)
            crop_mask[::d, ::d] = mask
            crop_mask = np.asarray(
                Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
                    (self.crop_width, self.crop_height),
                    Image.Resampling.NEAREST,
                )
            ) > 0
        else:
            crop_mask = mask
        resized = np.zeros((self.resized_height, self.resized_width), dtype=bool)
        resized[
            self.crop_top : self.crop_top + self.crop_height,
            self.crop_left : self.crop_left + self.crop_width,
        ] = crop_mask[: self.crop_height, : self.crop_width]
        return np.asarray(
            Image.fromarray(resized.astype(np.uint8) * 255).resize(
                (self.raw_width, self.raw_height),
                Image.Resampling.NEAREST,
            )
        ) > 0


@dataclass(frozen=True)
class FrontendImageViews:
    rgb_raw: np.ndarray
    rgb_sam3: np.ndarray
    transform: ImageTransform2D


def _resize_pil_image(img: Image.Image, long_edge_size: int) -> Image.Image:
    size = max(img.size)
    interp = Image.Resampling.LANCZOS if size > long_edge_size else Image.Resampling.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / size)) for x in img.size)
    return img.resize(new_size, interp)


def _as_uint8_rgb(rgb_raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_raw)
    if arr.dtype == np.uint8:
        out = arr
    else:
        out = np.uint8(np.clip(arr, 0.0, 1.0) * 255.0)
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError("rgb_raw must have shape (H, W, 3)")
    return out



def build_original_image_views(rgb_raw: np.ndarray) -> FrontendImageViews:
    """Build original-resolution frontend views with identity coordinates."""
    raw = _as_uint8_rgb(rgb_raw)
    raw_h, raw_w = raw.shape[:2]
    transform = ImageTransform2D(
        raw_height=int(raw_h),
        raw_width=int(raw_w),
        resized_height=int(raw_h),
        resized_width=int(raw_w),
        crop_left=0,
        crop_top=0,
        crop_width=int(raw_w),
        crop_height=int(raw_h),
        downsample=1,
        inference_height=int(raw_h),
        inference_width=int(raw_w),
    )
    return FrontendImageViews(rgb_raw=raw, rgb_sam3=raw.copy(), transform=transform)


def build_functional_slam_image_views(
    rgb_raw: np.ndarray,
    img_size: int,
    img_downsample: int,
    square_ok: bool = False,
) -> FrontendImageViews:
    """Build Functional-SLAM-compatible raw and SAM3 image views.

    This mirrors mast3r_slam.mast3r_utils.resize_img plus create_frame's
    uimg downsample path, without importing MASt3R/Dust3R model modules.
    """
    if int(img_size) not in {224, 512}:
        raise ValueError("img_size must be 224 or 512")
    downsample = max(1, int(img_downsample or 1))
    raw = _as_uint8_rgb(rgb_raw)
    raw_h, raw_w = raw.shape[:2]
    img = Image.fromarray(raw)
    w1, h1 = img.size
    if int(img_size) == 224:
        img = _resize_pil_image(img, round(int(img_size) * max(w1 / h1, h1 / w1)))
    else:
        img = _resize_pil_image(img, int(img_size))
    resized_w, resized_h = img.size
    cx, cy = resized_w // 2, resized_h // 2
    if int(img_size) == 224:
        half = min(cx, cy)
        crop_left = int(cx - half)
        crop_top = int(cy - half)
        crop_right = int(cx + half)
        crop_bottom = int(cy + half)
    else:
        halfw = int(((2 * cx) // 16) * 8)
        halfh = int(((2 * cy) // 16) * 8)
        if not square_ok and resized_w == resized_h:
            halfh = int(3 * halfw / 4)
        crop_left = int(cx - halfw)
        crop_top = int(cy - halfh)
        crop_right = int(cx + halfw)
        crop_bottom = int(cy + halfh)
    img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    sam3 = np.asarray(img)
    if downsample > 1:
        sam3 = sam3[::downsample, ::downsample]
    transform = ImageTransform2D(
        raw_height=int(raw_h),
        raw_width=int(raw_w),
        resized_height=int(resized_h),
        resized_width=int(resized_w),
        crop_left=int(crop_left),
        crop_top=int(crop_top),
        crop_width=int(crop_right - crop_left),
        crop_height=int(crop_bottom - crop_top),
        downsample=int(downsample),
        inference_height=int(sam3.shape[0]),
        inference_width=int(sam3.shape[1]),
    )
    return FrontendImageViews(rgb_raw=raw, rgb_sam3=sam3.astype(np.uint8), transform=transform)
