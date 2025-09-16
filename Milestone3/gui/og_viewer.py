"""
Interactive PyGame viewer for the occupancy grid (OG) and A* path.

Features:
- Renders the current OG built from GridMap as the main view.
- Highlights the current robot location in green (from EKF pose) each frame.
- Lets the user click to set an endpoint on the map; plans a path from current
  pose to the endpoint using A* and overlays it in blue.
- Designed to be extensible for additional overlays and info panes.

This module does not alter the underlying OG values; all drawings are overlays
on a rendered copy per frame.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple, List

import numpy as np
import pygame
import cv2

# Import from sibling modules when launched as a script from Milestone3
from planning.grid_map import GridMap
from planning.astar import AStarPlanner, PlanResult


class OGViewer:
    """PyGame-based interactive viewer for OG and planning."""

    def __init__(
        self,
        grid: GridMap,
        planner: Optional[AStarPlanner] = None,
        get_pose_fn=None,
        window_scale: int = 4,
        fps: int = 15,
    ) -> None:
        """
        Parameters
        - grid: GridMap instance to visualise
        - planner: AStarPlanner for computing a path (optional but recommended)
        - get_pose_fn: callable that returns [x, y, theta] (EKF pose). If None,
          the pose defaults to (0,0,0).
        - window_scale: integer scale factor used for rendering the OG
        - fps: frames per second for the main loop
        """
        self.grid = grid
        self.planner = planner or AStarPlanner()
        self.get_pose_fn = get_pose_fn
        self.scale = int(max(1, window_scale))
        self.fps = int(max(1, fps))

        # Interactive state
        self.goal_xy: Optional[Tuple[float, float]] = None
        self.last_plan: Optional[PlanResult] = None

        # Init PyGame
        pygame.init()
        pygame.display.set_caption("Occupancy Grid Viewer")

        # Pre-render initial OG image to determine window size
        self._vis = self.grid.render(scale=self.scale)
        h, w = self._vis.shape[:2]
        self._surface = pygame.display.set_mode((w, h))
        self._clock = pygame.time.Clock()

    def _draw_overlay(self, surf: pygame.Surface) -> None:
        """Draw robot pose (green), goal (red), and path (blue) overlays on the surface."""
        # Convert numpy OG image to PyGame surface each frame (copy for overlay)
        vis_bgr = self.grid.render(scale=self.scale)

        # Draw robot pose as a green dot
        pose = [0.0, 0.0, 0.0]
        if callable(self.get_pose_fn):
            try:
                pose = self.get_pose_fn()
            except Exception:
                pass
        r_r, c_r = self.grid.world_to_grid(float(pose[0]), float(pose[1]))
        cv2.circle(vis_bgr, (int(c_r * self.scale), int(r_r * self.scale)), 5, (0, 200, 0), -1)

        # Draw goal in red
        if self.goal_xy is not None:
            r_g, c_g = self.grid.world_to_grid(*self.goal_xy)
            cv2.circle(vis_bgr, (int(c_g * self.scale), int(r_g * self.scale)), 5, (0, 0, 220), -1)

        # Draw last planned path in blue (if available)
        if self.last_plan is not None and self.last_plan.path_grid:
            pts = np.array([[int(c * self.scale), int(r * self.scale)] for (r, c) in self.last_plan.path_grid], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis_bgr, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

        # Convert BGR numpy image to PyGame surface and blit
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        pygame_surf = pygame.surfarray.make_surface(np.rot90(vis_rgb))
        pygame_surf = pygame.transform.flip(pygame_surf, True, False)
        surf.blit(pygame_surf, (0, 0))

    def _pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert mouse pixel to world (x, y) using grid mapping and scale."""
        # Reverse of how we draw: our OG is drawn with row->y, col->x scaled
        c = int(px / self.scale)
        r = int(py / self.scale)
        x, y = self.grid.grid_to_world(r, c)
        return x, y

    def _replan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> None:
        """Run the A* planner and store the result for overlay."""
        try:
            self.last_plan = self.planner.plan(self.grid, start_xy, goal_xy)
            if self.last_plan is None:
                print("[OGViewer] No path found.")
            else:
                print(f"[OGViewer] Planned path with {len(self.last_plan.path_grid)} cells, cost {self.last_plan.cost:.2f}")
        except Exception as e:
            print("[OGViewer] Planning error:", e)
            self.last_plan = None

    def run(self) -> None:
        """Main interactive loop. Left-click to set a new goal and plan path."""
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Left-click sets goal and triggers replan
                    mx, my = pygame.mouse.get_pos()
                    gx, gy = self._pixel_to_world(mx, my)
                    self.goal_xy = (gx, gy)
                    # Use current pose as start
                    pose = [0.0, 0.0, 0.0]
                    if callable(self.get_pose_fn):
                        try:
                            pose = self.get_pose_fn()
                        except Exception:
                            pass
                    start_xy = (float(pose[0]), float(pose[1]))
                    self._replan(start_xy, self.goal_xy)

            # Draw overlays on OG and update display
            self._surface.fill((0, 0, 0))
            self._draw_overlay(self._surface)
            pygame.display.flip()
            self._clock.tick(self.fps)

        pygame.quit()
