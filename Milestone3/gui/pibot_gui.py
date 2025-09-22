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
import json
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
from state_machine.state_machine import PiBotFruitSearchSM
from util.pibot import PenguinPi

log = logging.getLogger(__name__)


class PiBotGUI:
    """PyGame-based interactive viewer for OG and planning."""

    def __init__(
        self,
        # Import Instances of objects
        grid: GridMap,
        ppi: PenguinPi,
        planner: AStarPlanner,
        state_machine: PiBotFruitSearchSM,
        detector=None,
        fruit_ranger=None,
        # Controller type
        controller_kind: str = "ttg",
        # Import functions
        get_pose_fn = None,
        get_frame_fn = None,
        # Gui Settings
        window_scale: int = 4,
        fps: int = 15,  
        dry_run: bool = False,
        ARUCO_locations: np.ndarray = None,
        target_dims: Optional[dict] = None,
        # Display-only hooks
        interactive: bool = False,
        intent_sink=None,
        plan_provider=None,
        detections_provider=None,
        status_provider=None,
        # Intent helpers for runtime control
        mode_sink=None,
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
        self.planner = planner
        self.state_machine = state_machine
        self.get_pose_fn = get_pose_fn
        self.scale = int(max(1, window_scale))
        self.fps = int(max(1, fps))
        self.ctrl = ControllerManager(controller_kind)
        self.dry_run = bool(dry_run)
        self.ppi = ppi

        self.ARUCO_locations = ARUCO_locations
        # Live detection handles
        self.get_frame_fn = get_frame_fn
        self.detector = detector
        self.fruit_ranger = fruit_ranger
        self.target_dims = target_dims or {}
        # Display-only providers / intents
        self.interactive = bool(interactive)
        self.intent_sink = intent_sink
        self.plan_provider = plan_provider
        self.detections_provider = detections_provider
        self.status_provider = status_provider
        self.mode_sink = mode_sink

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
        # Map panel size
        map_w, map_h = w, h
        # If a camera frame source is provided, allocate a right-hand panel of same size
        self._has_cam = callable(self.get_frame_fn)
        win_w = map_w * (2 if self._has_cam else 1)
        self._surface = pygame.display.set_mode((win_w, map_h))
        self._clock = pygame.time.Clock()
        log.info("OGViewer init: window=%dx%d scale=%d fps=%d", w, h, self.scale, self.fps)

    def _draw_overlay(self, surf: pygame.Surface, pose_xyz: Tuple[float, float, float]) -> None:
        """Draw start (green), current (blue), goal (red), path (blue), and robot sprite."""
        # Convert numpy OG image to PyGame surface each frame (copy for overlay)
        vis_bgr = self.grid.render(scale=self.scale)

        # Start in green
        if self.start_xy is not None:
            r_s, c_s = self.grid.world_to_grid(*self.start_xy)
            cv2.circle(vis_bgr, (int(c_s * self.scale), int(r_s * self.scale)), 6, (0, 200, 0), -1)

        # Current in blue/orange and robot sprite
        rx, ry, rth = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
        r_r, c_r = self.grid.world_to_grid(rx, ry)
        px, py = int(c_r * self.scale), int(r_r * self.scale)
        cv2.circle(vis_bgr, (px, py), 5, (220, 140, 0), -1)

        # Draw robot top sprite rotated to heading if available
        try:
            if not hasattr(self, '_pibot_icon'):
                self._pibot_icon = cv2.imread('pics/8bit/pibot_top.png', cv2.IMREAD_UNCHANGED)
            icon = getattr(self, '_pibot_icon', None)
            if icon is not None and icon.shape[2] == 4:
                # Resize to target size
                target_sz = 24
                icon_rsz = cv2.resize(icon, (target_sz, target_sz), interpolation=cv2.INTER_AREA)
                # Compute rotation: sprite points left initially; world heading 0 points +x
                angle_deg = 180.0 - (-rth * 180.0 / np.pi)
                # Rotate with bounding box
                M = cv2.getRotationMatrix2D((target_sz / 2.0, target_sz / 2.0), angle_deg, 1.0)
                cos = abs(M[0, 0]); sin = abs(M[0, 1])
                nW = int((target_sz * sin) + (target_sz * cos))
                nH = int((target_sz * cos) + (target_sz * sin))
                M[0, 2] += (nW / 2.0) - (target_sz / 2.0)
                M[1, 2] += (nH / 2.0) - (target_sz / 2.0)
                icon_rot = cv2.warpAffine(icon_rsz, M, (nW, nH), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                ih, iw = icon_rot.shape[:2]
                x0 = int(px - iw // 2); y0 = int(py - ih // 2)
                x1 = x0 + iw; y1 = y0 + ih
                H, W = vis_bgr.shape[:2]
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(W, x1), min(H, y1)
                if x1c > x0c and y1c > y0c:
                    roi = vis_bgr[y0c:y1c, x0c:x1c]
                    ix0, iy0 = x0c - x0, y0c - y0
                    ix1, iy1 = ix0 + (x1c - x0c), iy0 + (y1c - y0c)
                    icon_crop = icon_rot[iy0:iy1, ix0:ix1]
                    alpha = icon_crop[:, :, 3:] / 255.0
                    roi[:] = (1 - alpha) * roi + alpha * icon_crop[:, :, :3]
        except Exception:
            pass

        # Draw goal in red
        if self.goal_xy is not None:
            r_g, c_g = self.grid.world_to_grid(*self.goal_xy)
            cv2.circle(vis_bgr, (int(c_g * self.scale), int(r_g * self.scale)), 5, (0, 0, 220), -1)

        # Draw provided plan (preferred) or last planned path
        drew_provider = False
        if callable(self.plan_provider):
            try:
                plan = self.plan_provider() or None
            except Exception:
                plan = None
            if isinstance(plan, dict) and plan.get('waypoints'):
                pts = []
                for (wx, wy) in plan['waypoints']:
                    r_p, c_p = self.grid.world_to_grid(float(wx), float(wy))
                    pts.append([int(c_p * self.scale), int(r_p * self.scale)])
                if len(pts) >= 2:
                    cv2.polylines(vis_bgr, [np.array(pts, dtype=np.int32)], isClosed=False, color=(50, 120, 225), thickness=2)
                    drew_provider = True
        if not drew_provider and self.last_plan is not None and self.last_plan.path_grid:
            pts = np.array([[int(c * self.scale), int(r * self.scale)] for (r, c) in self.last_plan.path_grid], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(vis_bgr, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

        # Overlay ARUCO marker sprites at their OG positions using self.ARUCO_locations (Nx2 array)
        try:
            if self.ARUCO_locations is not None and hasattr(self.ARUCO_locations, 'shape') and self.ARUCO_locations.shape[1] == 3:
                #log.debug(f"Loading {self.ARUCO_locations.shape[0]} ARUCO sprites")
                for tag in self.ARUCO_locations:
                    tag_id = int(tag[2])
                    # Index mapping: index 0 -> aruco1, 9 -> aruco10
                    x, y = float(tag[0]), float(tag[1])
                    r, c = self.grid.world_to_grid(x, y)
                    px, py = int(c * self.scale), int(r * self.scale)
                    # Load correct sprite for this tag id
                    icon_path = f'pics/8bit/lm_{tag_id}.png'
                    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                    if icon is None:
                        # Fallback placeholder box with id label
                        sz = 14
                        tl = (px - sz // 2, py - sz // 2)
                        br = (px + sz // 2, py + sz // 2)
                        cv2.rectangle(vis_bgr, tl, br, (0, 0, 255), 2)
                        cv2.putText(vis_bgr, f"{tag_id}", (tl[0], tl[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                        continue
                    # Resize sprite to a sensible size
                    target_sz = 18
                    icon_rsz = cv2.resize(icon, (target_sz, target_sz), interpolation=cv2.INTER_AREA)
                    iw, ih = icon_rsz.shape[1], icon_rsz.shape[0]
                    x0 = px - iw // 2
                    y0 = py - ih // 2
                    x1 = x0 + iw
                    y1 = y0 + ih
                    # Clip to bounds
                    H, W = vis_bgr.shape[:2]
                    x0c, y0c = max(0, x0), max(0, y0)
                    x1c, y1c = min(W, x1), min(H, y1)
                    if x1c <= x0c or y1c <= y0c:
                        continue
                    roi = vis_bgr[y0c:y1c, x0c:x1c]
                    ix0, iy0 = x0c - x0, y0c - y0
                    ix1, iy1 = ix0 + (x1c - x0c), iy0 + (y1c - y0c)
                    icon_crop = icon_rsz[iy0:iy1, ix0:ix1]
                    if icon_crop.shape[2] == 4:
                        alpha = icon_crop[:, :, 3:] / 255.0
                        roi[:] = (1 - alpha) * roi + alpha * icon_crop[:, :, :3]
                    else:
                        roi[:] = icon_crop
        except Exception as e:
            log.debug("ARUCO sprite overlay failed: %s", e)

        # Status overlay (mode / SM state / action)
        try:
            if callable(self.status_provider):
                st = self.status_provider() or {}
                txt = f"Mode:{st.get('mode','')} SM:{st.get('sm_state','')} Act:{st.get('action','')} {st.get('progress','')}"
                cv2.putText(vis_bgr, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_bgr, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

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

    def _replan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> bool:
        """Run the A* planner and, if successful, replace the current plan.

        Returns True on success, False if planning failed (previous plan retained).
        """
        try:
            log.info("Planning from %s to %s", str(start_xy), str(goal_xy))
            new_plan = self.planner.plan(self.grid, start_xy, goal_xy)
            if new_plan is None:
                log.warning("No path found; retaining previous plan")
                return False
            self.last_plan = new_plan
            self._wp_idx = 0
            self._last_plan_time = time.time()
            log.info("Planned path cells=%d cost=%.2f", len(self.last_plan.path_grid), self.last_plan.cost)
            return True
        except Exception as e:
            log.exception("Planning error: %s; retaining previous plan", e)
            return False

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
        # Only compute cross-track if we still have a valid plan
        if self.last_plan is not None:
            xtrack = AStarPlanner.cross_track_error((pose[0], pose[1]), self.last_plan.pruned_world)
            if xtrack > self._xtrack_thresh:
                log.info("Cross-track error %.3f > %.3f; attempting replanning", xtrack, self._xtrack_thresh)
                self.start_xy = (float(pose[0]), float(pose[1]))
                ok = self._replan(self.start_xy, self.goal_xy)
                # If replanning failed, continue following previous plan; do not early return

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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    # Start state machine (switch to AUTO mode)
                    if callable(self.mode_sink):
                        try:
                            self.mode_sink('AUTO')
                        except Exception:
                            pass
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.interactive:
                    # Left-click: emit intent if sink provided; else plan locally as before
                    mx, my = pygame.mouse.get_pos()
                    gx, gy = self._pixel_to_world(mx, my)
                    log.info("Mouse click -> goal=(%.3f, %.3f)", gx, gy)
                    if callable(self.intent_sink):
                        try:
                            self.intent_sink(gx, gy)
                        except Exception:
                            pass
                    else:
                        self.goal_xy = (gx, gy)
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
            self._draw_overlay(self._surface, (float(pose[0]), float(pose[1]), float(pose[2])))

            # Right panel: live video + detector results
            if self._has_cam:
                try:
                    frame_rgb = self.get_frame_fn()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    sys.stderr.write(f"camera_error: {e}\n")
                    frame_bgr = np.zeros_like(self._vis)

                det_vis = frame_bgr.copy()
                results = []
                try:
                    if self.detector is not None:
                        det_out, det_vis = self.detector.detect_single_image(frame_bgr)
                        if isinstance(det_out, list):
                            for item in det_out:
                                # Expect (label, [x,y,w,h])
                                try:
                                    label, bbox = item[0], item[1]
                                except Exception:
                                    continue
                                # Compute range/theta with FruitRanger if available
                                rng = -1.0; th = 0.0
                                if self.fruit_ranger is not None:
                                    true_h = None
                                    if isinstance(label, str) and label in self.target_dims:
                                        dims = self.target_dims[label]
                                        if isinstance(dims, (list, tuple)) and len(dims) == 3:
                                            true_h = float(dims[2])
                                    if true_h is None:
                                        true_h = 0.08
                                    # Select your method!
                                    est = self.fruit_ranger.from_bbox_height(bbox, true_h)
                                    #est = self.fruit_ranger.from_ground_ray(bbox)
                                    if est is not None:
                                        rng = float(est['r'])
                                        th = float(np.degrees(est['theta']))
                                # class_id mapping
                                if isinstance(label, int):
                                    cid = int(label)
                                else:
                                    keys = list(self.target_dims.keys())
                                    cid = keys.index(label) if label in keys else -1
                                results.append({"class_id": cid, "range": rng, "theta": th})
                    # Print JSON line for this frame
                    try:
                        log.debug(json.dumps(results) + "\n")
                    except Exception as e:
                        log.debug(f"json_error: {e}\n")
                except Exception as e:
                    log.error(f"detector_error: {e}\n")

                # Draw the camera panel on the right
                try:
                    mh, mw = self._vis.shape[:2]
                    cam_panel = cv2.resize(det_vis, (mw, mh), interpolation=cv2.INTER_NEAREST)
                    cam_rgb = cv2.cvtColor(cam_panel, cv2.COLOR_BGR2RGB)
                    pygame_cam = pygame.surfarray.make_surface(np.rot90(cam_rgb))
                    pygame_cam = pygame.transform.flip(pygame_cam, True, False)
                    self._surface.blit(pygame_cam, (mw, 0))
                except Exception as e:
                    sys.stderr.write(f"cam_panel_error: {e}\n")

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
