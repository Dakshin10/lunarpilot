"""
High-Resolution Resizable Pygame Display & Aerospace Telemetry HUD for LunarPilot.
Upscales Gymnasium's default 600x400 viewport to 1200x800 (or custom resolutions)
with dynamic window resizing and real-time flight telemetry overlay.
"""

import gymnasium as gym
import numpy as np


class ScaledRenderWrapper(gym.Wrapper):
    """
    Wraps Gymnasium's LunarLander environment to upscale the Pygame window,
    support real-time window resizing, and project a heads-up display (HUD)
    with avionics telemetry.
    """

    def __init__(
        self,
        env: gym.Env,
        scale: float = 2.0,
        width: int = None,
        height: int = None,
        resizable: bool = True,
        enable_hud: bool = True,
        window_title: str = "LunarPilot - Autonomous Flight Display (Continuous PPO)",
    ):
        super().__init__(env)
        self.scale = scale
        self.target_w = width if width is not None else int(600 * scale)
        self.target_h = height if height is not None else int(400 * scale)
        self.resizable = resizable
        self.enable_hud = enable_hud
        self.window_title = window_title

        self._screen = None
        self._font = None
        self._font_small = None
        self._last_obs = None
        self._last_info = {}
        self.should_exit = False

        if getattr(self.env.unwrapped, "render_mode", None) == "human":
            self._setup_custom_render()

    def _setup_custom_render(self):
        try:
            import pygame
        except ImportError:
            return

        u = self.env.unwrapped
        self._orig_render = u.render

        def custom_render():
            if self._screen is None:
                pygame.display.init()
                pygame.font.init()
                flags = pygame.RESIZABLE if self.resizable else 0
                self._screen = pygame.display.set_mode(
                    (self.target_w, self.target_h), flags
                )
                pygame.display.set_caption(self.window_title)

                # Initialize monospace telemetry fonts
                try:
                    self._font = pygame.font.SysFont("Consolas", 14, bold=True)
                    self._font_small = pygame.font.SysFont("Consolas", 12, bold=True)
                except Exception:
                    self._font = pygame.font.Font(None, 18)
                    self._font_small = pygame.font.Font(None, 14)

            # Prevent original render from blitting fixed 600x400 to the real screen
            u.screen = pygame.Surface((600, 400))
            self._orig_render()
            u.screen = self._screen

            # Event pump: handle window resizing, close button, and ESC/q key
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.should_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self.should_exit = True
                elif event.type == pygame.VIDEORESIZE:
                    self.target_w, self.target_h = max(400, event.w), max(300, event.h)
                    self._screen = pygame.display.set_mode(
                        (self.target_w, self.target_h), pygame.RESIZABLE
                    )
                    u.screen = self._screen

            # Smoothly upscale the 600x400 surface to current window resolution
            scaled_surf = pygame.transform.smoothscale(
                u.surf, (self.target_w, self.target_h)
            )
            self._screen.blit(scaled_surf, (0, 0))

            # Render real-time avionics HUD overlay
            if self.enable_hud and self._last_obs is not None:
                self._draw_hud(self._screen)

            pygame.display.flip()

        u.render = custom_render

    def _draw_hud(self, surface):
        import pygame

        obs = self._last_obs
        x, y, vx, vy, angle, omega, leg_l, leg_r = obs

        hud_h = 36
        banner = pygame.Surface((self.target_w, hud_h), pygame.SRCALPHA)
        banner.fill((10, 16, 26, 215))  # Aerospace dark blue glass
        surface.blit(banner, (0, 0))
        pygame.draw.line(surface, (0, 180, 255), (0, hud_h), (self.target_w, hud_h), 1)

        speed = float(np.sqrt(vx**2 + vy**2))
        pitch_deg = float(np.degrees(angle))
        fuel_pct = float(self._last_info.get("fuel_pct", 100.0))
        flameout = self._last_info.get("flameout", False)

        # Status indicator
        if flameout:
            status_text = "FLAMEOUT"
            status_color = (255, 60, 60)
        elif leg_l == 1.0 and leg_r == 1.0:
            status_text = "TOUCHDOWN"
            status_color = (46, 204, 113)
        elif y <= 0.15:
            status_text = "TERMINAL FLARE"
            status_color = (241, 196, 15)
        else:
            status_text = "GUIDED DESCENT"
            status_color = (0, 230, 255)

        gear_str = f"[{'L' if leg_l else '-'}{'R' if leg_r else '-'}]"

        if self.target_w >= 900:
            hud_items = [
                f"SYS: {status_text}",
                f"ALT: {y:+.2f}m",
                f"PAD-X: {x:+.2f}m",
                f"V-SINK: {vy:+.2f}m/s",
                f"H-DRIFT: {vx:+.2f}m/s",
                f"SPEED: {speed:.2f}m/s",
                f"PITCH: {pitch_deg:+.1f}°",
                f"FUEL: {fuel_pct:.0f}%",
                f"GEAR: {gear_str}",
            ]
            full_text = "  |  ".join(hud_items)
            text_surf = self._font.render(full_text, True, (220, 240, 255))
        else:
            compact_items = [
                f"{status_text}",
                f"ALT:{y:+.1f}m",
                f"VY:{vy:+.2f}",
                f"VX:{vx:+.2f}",
                f"θ:{pitch_deg:+.0f}°",
                f"FUEL:{fuel_pct:.0f}%",
            ]
            full_text = " | ".join(compact_items)
            text_surf = self._font_small.render(full_text, True, (220, 240, 255))

        text_rect = text_surf.get_rect(center=(self.target_w // 2, hud_h // 2))
        surface.blit(text_surf, text_rect)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self._last_info = info
        if self.should_exit:
            terminated = True
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.env, name)
