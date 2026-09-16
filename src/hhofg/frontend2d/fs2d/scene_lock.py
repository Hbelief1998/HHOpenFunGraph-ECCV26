from __future__ import annotations

# 本模块提供场景类型锁定器：在连续高置信预测后锁定场景，避免抖动。

from typing import Optional


class SceneLock:
    """Locks scene_type after M consecutive high-confidence consistent predictions."""

    def __init__(
        self,
        m: int = 3,
        conf_thresh: float = 0.8,
        switch_m: int = 2,
        switch_conf_thresh: float | None = None,
        switch_cooldown_frames: int = 20,
    ) -> None:
        # 连续命中次数阈值与置信度阈值。
        self.m = int(m)
        self.conf_thresh = float(conf_thresh)
        # locked_scene 一旦设置，将不再被更新。
        self.locked_scene: Optional[str] = None
        # 用于记录当前连续命中的场景与次数。
        self.streak_scene: Optional[str] = None
        self.streak = 0
        # 已锁定后的低频场景切换判定，不影响 update() 的初始锁定语义。
        self.switch_m = int(switch_m)
        self.switch_conf_thresh = float(switch_conf_thresh if switch_conf_thresh is not None else conf_thresh)
        self.switch_cooldown_frames = int(switch_cooldown_frames)
        self.switch_candidate_scene: Optional[str] = None
        self.switch_streak = 0
        self.last_switch_frame: Optional[int] = None

    def update(self, scene_type: str, confidence: float) -> dict:
        # 如果已锁定，直接返回锁定结果。
        if self.locked_scene is not None:
            return {
                "locked": True,
                "scene_type": self.locked_scene,
                "streak": self.streak,
                "reason": "already_locked",
            }

        if scene_type is None or scene_type == "unknown":
            self.streak_scene = None
            self.streak = 0
            return {
                "locked": False,
                "scene_type": None,
                "streak": 0,
                "reason": "unknown_scene",
            }

        # 置信度不足时重置 streak，避免低质量预测影响锁定。
        if confidence < self.conf_thresh:
            self.streak_scene = None
            self.streak = 0
            return {
                "locked": False,
                "scene_type": None,
                "streak": 0,
                "reason": "below_conf_thresh",
            }

        # 统计连续一致的场景预测次数。
        if self.streak_scene == scene_type:
            self.streak += 1
        else:
            self.streak_scene = scene_type
            self.streak = 1

        # 达到门限后锁定场景类型。
        if self.streak >= self.m:
            self.locked_scene = scene_type
            return {
                "locked": True,
                "scene_type": self.locked_scene,
                "streak": self.streak,
                "reason": "locked",
            }

        return {
            "locked": False,
            "scene_type": None,
            "streak": self.streak,
            "reason": "streaking",
        }

    def _reset_switch_candidate(self) -> None:
        self.switch_candidate_scene = None
        self.switch_streak = 0

    def consider_switch(self, scene_type: str | None, confidence: float, frame_idx: int) -> dict:
        """Consider switching an already locked scene using hysteresis."""
        if self.locked_scene is None:
            return {
                "switched": False,
                "scene_type": None,
                "old_scene": None,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "not_locked",
            }

        if scene_type is None or scene_type == "unknown":
            self._reset_switch_candidate()
            return {
                "switched": False,
                "scene_type": self.locked_scene,
                "old_scene": None,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "unknown_scene",
            }

        if confidence < self.switch_conf_thresh:
            self._reset_switch_candidate()
            return {
                "switched": False,
                "scene_type": self.locked_scene,
                "old_scene": None,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "below_switch_conf_thresh",
            }

        if scene_type == self.locked_scene:
            self._reset_switch_candidate()
            return {
                "switched": False,
                "scene_type": self.locked_scene,
                "old_scene": None,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "same_scene",
            }

        if (
            self.last_switch_frame is not None
            and int(frame_idx) - int(self.last_switch_frame) < self.switch_cooldown_frames
        ):
            self._reset_switch_candidate()
            return {
                "switched": False,
                "scene_type": self.locked_scene,
                "old_scene": None,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "cooldown",
            }

        if scene_type == self.switch_candidate_scene:
            self.switch_streak += 1
        else:
            self.switch_candidate_scene = scene_type
            self.switch_streak = 1

        if self.switch_streak >= self.switch_m:
            old_scene = self.locked_scene
            self.locked_scene = scene_type
            self.last_switch_frame = int(frame_idx)
            self._reset_switch_candidate()
            return {
                "switched": True,
                "scene_type": self.locked_scene,
                "old_scene": old_scene,
                "candidate_scene": None,
                "switch_streak": 0,
                "reason": "switched",
            }

        return {
            "switched": False,
            "scene_type": self.locked_scene,
            "old_scene": None,
            "candidate_scene": scene_type,
            "switch_streak": self.switch_streak,
            "reason": "switch_streaking",
        }
