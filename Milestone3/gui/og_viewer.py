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
import logging

import numpy as np
import pygame
import cv2

# Debugger
import pdb;

# Import from sibling modules when launched as a script from Milestone3
from planning.grid_map import GridMap
from planning.astar import AStarPlanner, PlanResult
from navigation.controller import ControllerManager
log = logging.getLogger(__name__)


class OGViewer:
    """PyGame-based interactive viewer for OG and planning."""

    def __init__(
        self,
        grid: GridMap,
        planner: Optional[AStarPlanner] = None,
        get_pose_fn=None,
        window_scale: int = 4,
        fps: int = 15,
        controller_kind: str = "ttg",
        dry_run: bool = False,
        ppi=None,
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
        self.ctrl = ControllerManager(controller_kind)
        self.dry_run = bool(dry_run)
        self.ppi = ppi

        # Interactive state
        self.goal_xy: Optional[Tuple[float, float]] = None
        self.start_xy: Optional[Tuple[float, float]] = None
        self.last_plan: Optional[PlanResult] = None
        self._wp_idx: int = 0
        self._last_plan_time: float = 0.0
        self._xtrack_thresh: float = 0.15

        # Init PyGame
        pygame.init()
        pygame.display.set_caption("Occupancy Grid Viewer")

        # Pre-render initial OG image to determine window size
        self._vis = self.grid.render(scale=self.scale)
        h, w = self._vis.shape[:2]
        self._surface = pygame.display.set_mode((w, h))
        self._clock = pygame.time.Clock()
        log.info("OGViewer init: window=%dx%d scale=%d fps=%d", w, h, self.scale, self.fps)

    def _draw_overlay(self, surf: pygame.Surface, pose_xy: Tuple[float, float]) -> None:
        """Draw start (green), current (blue), goal (red), and path (blue)."""
        # Convert numpy OG image to PyGame surface each frame (copy for overlay)
        vis_bgr = self.grid.render(scale=self.scale)

        # Start in green
        if self.start_xy is not None:
            r_s, c_s = self.grid.world_to_grid(*self.start_xy)
            cv2.circle(vis_bgr, (int(c_s * self.scale), int(r_s * self.scale)), 6, (0, 200, 0), -1)

        # Current in blue/orange
        r_r, c_r = self.grid.world_to_grid(float(pose_xy[0]), float(pose_xy[1]))
        cv2.circle(vis_bgr, (int(c_r * self.scale), int(r_r * self.scale)), 5, (220, 140, 0), -1)

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
            log.info("Planning from %s to %s", str(start_xy), str(goal_xy))
            self.last_plan = self.planner.plan(self.grid, start_xy, goal_xy)
            if self.last_plan is None:
                log.warning("No path found")
            else:
                self._wp_idx = 0
                self._last_plan_time = time.time()
                log.info("Planned path cells=%d cost=%.2f", len(self.last_plan.path_grid), self.last_plan.cost)
        except Exception as e:
            log.exception("Planning error: %s", e)
            self.last_plan = None

    def _control_step(self, pose: Tuple[float, float, float]) -> None:
        if self.goal_xy is None or self.last_plan is None or not self.last_plan.pruned_world:
            if not self.dry_run and self.ppi is not None:
                self.ppi.set_velocity([0, 0])
            return

        # Periodic replan
        if AStarPlanner.time_to_replan(self._last_plan_time, period_s=2.5):
            self.start_xy = (float(pose[0]), float(pose[1]))
            self._replan(self.start_xy, self.goal_xy)

        # Cross-track replan
        xtrack = AStarPlanner.cross_track_error((pose[0], pose[1]), self.last_plan.pruned_world)
        if xtrack > self._xtrack_thresh:
            log.info("Cross-track error %.3f > %.3f; replanning", xtrack, self._xtrack_thresh)
            self.start_xy = (float(pose[0]), float(pose[1]))
            self._replan(self.start_xy, self.goal_xy)
            return

        # Control towards current waypoint
        self._wp_idx = min(self._wp_idx, len(self.last_plan.pruned_world) - 1)
        wp = self.last_plan.pruned_world[self._wp_idx]
        fwd_cmd, turn_cmd, fwd_tick, turn_tick, done = self.ctrl.compute(pose, wp)
        if not self.dry_run and self.ppi is not None:
            log.debug("Control: fwd_cmd=%d turn_cmd=%d fwd_tick=%d turn_tick=%d", fwd_cmd, turn_cmd, fwd_tick, turn_tick)
            self.ppi.set_velocity([fwd_cmd, turn_cmd], tick=fwd_tick, turning_tick=turn_tick)

        if done:
            if self._wp_idx < len(self.last_plan.pruned_world) - 1:
                self._wp_idx += 1
            else:
                if not self.dry_run and self.ppi is not None:
                    log.info("Arrived at goal; stopping robot")
                    self.ppi.set_velocity([0, 0])

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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    pdb.set_trace()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Left-click sets goal and triggers replan
                    mx, my = pygame.mouse.get_pos()
                    gx, gy = self._pixel_to_world(mx, my)
                    self.goal_xy = (gx, gy)
                    log.info("Mouse click -> goal=(%.3f, %.3f)", gx, gy)
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
            pose = [0.0, 0.0, 0.0]
            
            if callable(self.get_pose_fn):
                try:
                    pose = self.get_pose_fn()
                except Exception:
                    pass
            self._draw_overlay(self._surface, (float(pose[0]), float(pose[1])))

            # Control step
            self._control_step((float(pose[0]), float(pose[1]), float(pose[2])))
            pygame.display.flip()
            self._clock.tick(self.fps)

        # Stop robot on exit
        if not self.dry_run and self.ppi is not None:
            try:
                log.info("Viewer closed; issuing stop to robot")
                self.ppi.set_velocity([0, 0])
            except Exception:
                pass
        pygame.quit()
