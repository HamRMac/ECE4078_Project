Object Ownership And Instantiation Hierarchy

- Main Orchestrator: auto_fruit_search.py
  - Creates `PenguinPi` ppi and passes to GUI and motion helpers.
    - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:487`
    - Passed to: `PiBotGUI(ppi=ppi, get_frame_fn=ppi.get_image)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:561`
  - Builds `GridMap` grid from ArUco markers and passes to GUI.
    - New: `GridMap(...)` then `grid.build_from_aruco(...)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:502`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:505`
    - Passed to: `PiBotGUI(grid=grid)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:553`
  - Creates `EKF` ekf which OWNS a `Robot` (constructed in `init_ekf`).
    - `Robot(baseline, scale, camera_matrix, dist_coeffs)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:252`
    - `EKF(robot)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:253`
  - Creates `aruco_detector` aruco_det using `ekf.robot` (consumes camera params from `Robot`).
    - `aruco.aruco_detector(ekf.robot, marker_length=0.07)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:520`
  - Optionally creates live perception helpers and passes to GUI:
    - `FruitRanger(camera_matrix=ekf.robot.camera_matrix)`
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:532`
      - Passed to: `PiBotGUI(fruit_ranger=fruit_ranger, target_dims=TARGET_DIMS)`
        - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:564`
    - `Detector(args.model, 384)` (YOLO)
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:537`
      - Passed to: `PiBotGUI(detector=yolo_detector)`
        - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:563`
  - Creates `AStarPlanner()` and passes to GUI.
    - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:554`
  - Creates `PiBotGUI` viewer which then owns its internal `ControllerManager` and drives robot via `ppi`.
    - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:553`
  - Runtime pose flow (used by GUI and navigation):
    - `get_robot_pose(ppi, aruco_det, ekf)` calls `ppi.get_image()` → `aruco_det.detect_marker_positions(img)` → `ekf.predict(Drive)` and `ekf.update(lms)` → returns `ekf.robot.state`.
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:175`

- GUI: gui/pibot_gui.py
  - Class: `PiBotGUI`
    - Receives: `grid: GridMap`, `planner: AStarPlanner`, `get_pose_fn`, `ppi`, `detector`, `fruit_ranger`, `target_dims`.
      - File: `ECE4078_Project/Milestone3/gui/pibot_gui.py:40`
    - Instantiates `ControllerManager(controller_kind)` internally (OWNS controller instance).
      - File: `ECE4078_Project/Milestone3/gui/pibot_gui.py:71`
    - Uses `planner.plan(grid, start, goal)` to compute path; stores `PlanResult`.
      - File: `ECE4078_Project/Milestone3/gui/pibot_gui.py:116` (overlay draw) and `:141` (replan helper)
    - Calls `self.ctrl.compute(pose, waypoint)` and commands robot via `ppi.set_velocity(...)`.
      - File: `ECE4078_Project/Milestone3/gui/pibot_gui.py:189`
    - If `detector`/`fruit_ranger` provided, runs detection on right-hand panel and estimates range/theta per detection.
      - File: `ECE4078_Project/Milestone3/gui/pibot_gui.py:214`

- Planning: planning/
  - Class: `GridMap`
    - Built in main; GUI holds a reference for rendering and world→grid mapping.
      - File: `ECE4078_Project/Milestone3/planning/grid_map.py:8`
  - Class: `AStarPlanner`
    - Stateless planner used by GUI; returns `PlanResult` dataclass.
      - File: `ECE4078_Project/Milestone3/planning/astar.py:26`
      - `PlanResult` dataclass defined at
        - File: `ECE4078_Project/Milestone3/planning/astar.py:17`

- Navigation Controllers: navigation/controller.py
  - Classes: `BaseController`, `TurnThenGoController`, `PurePursuitController`, `RHPController`, `ControllerManager`.
    - `ControllerManager(kind)` instantiates the chosen concrete controller and is OWNED by `PiBotGUI` (and also by `drive_to_point`).
      - File: `ECE4078_Project/Milestone3/navigation/controller.py:66`
    - In main ad-hoc nav: `ControllerManager(controller_kind)` used inside `drive_to_point` loop to compute commands.
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:146`

- Perception: YOLO + FruitRanger
  - Class: `Detector` (YOLO)
    - Instantiated optionally in main and passed to GUI; also used in other tools.
      - File: `ECE4078_Project/Milestone3/YOLO/detector.py:9`
  - Class: `FruitRanger`
    - Instantiated in main with EKF camera intrinsics and passed to GUI.
      - File: `ECE4078_Project/Milestone3/perception/fruit_ranger.py:27`
  - Class: `LiveTargetEstimator` (thread) — optional helper not wired into GUI by default.
    - Owns a `Detector` internally and pulls frames via provided callbacks.
      - File: `ECE4078_Project/Milestone3/auto_fruit_search.py:372`

- SLAM: slam/
  - Class: `Robot`
    - Created by `init_ekf` and OWNED by `EKF`; also passed to `aruco_detector` to supply camera params.
      - File: `ECE4078_Project/Milestone3/slam/robot.py:3`
  - Class: `EKF`
    - Created in main; OWNS `Robot`; consumes `measure.Drive` and `measure.Marker`.
      - File: `ECE4078_Project/Milestone3/slam/ekf.py:10`
  - Class: `aruco_detector`
    - Created in main with `ekf.robot`; returns `measure.Marker` list for `EKF.update`.
      - File: `ECE4078_Project/Milestone3/slam/aruco_detector.py:12`

- Robot I/O: util/
  - Class: `PenguinPi`
    - Created in main; OWNED by application; passed to GUI (for control and frames) and used by `get_robot_pose`.
      - File: `ECE4078_Project/Milestone3/util/pibot.py:10`
  - Classes: `measure.Drive`, `measure.Marker`
    - Constructed ad-hoc inside `get_robot_pose` and ArUco detector; consumed by `EKF.predict`/`EKF.update`.
      - File: `ECE4078_Project/Milestone3/util/measure.py:3`

- Actions Wrapper (not used in main): pibot_actions.py
  - Class: `PiBotActions`
    - OWNED by whichever script constructs it; wraps a `PenguinPi` instance (passed in) and optionally uses `Detector`/`FruitRanger` supplied to methods like `scan`.
      - File: `ECE4078_Project/Milestone3/pibot_actions.py:16`

- State Machine (standalone diagram generator): state_machine/state_machine.py
  - Class: `PiBotFruitSearchSM`
    - Instantiated as `robotSM` in module scope; used to emit a diagram; not integrated into `auto_fruit_search` run.
      - File: `ECE4078_Project/Milestone3/state_machine/state_machine.py:45`

Ownership Summary (who owns what)

- auto_fruit_search main owns: `PenguinPi`, `GridMap`, `EKF`→`Robot`, `aruco_detector`, optional `FruitRanger`, optional `Detector`, `AStarPlanner`, `PiBotGUI`.
- `PiBotGUI` owns: `ControllerManager` (and selected controller), current `PlanResult`, references to `GridMap`, `AStarPlanner`, `PenguinPi`, `Detector`, `FruitRanger`.
- `EKF` owns: `Robot`, landmark state/covariance.
- `aruco_detector` uses: `Robot` camera intrinsics; produces `measure.Marker` for `EKF`.
- `get_robot_pose` ties: `PenguinPi` → image, `aruco_detector` → markers, `EKF` → predict/update → pose.
- `PiBotActions` (if used) owns: a `PenguinPi` and drives it; pulls in `Detector`/`FruitRanger` when provided.

