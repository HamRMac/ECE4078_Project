from statemachine import StateMachine, State
from statemachine.contrib.diagram import DotGraphMachine

class PiBotFruitSearchSM(StateMachine):
    ## ALL STATES
    # Mapping states
    scan = State("Scan", initial=True)
    approach_fruit = State("ApproachFruit")
    sit_next_to_close_fruit = State("SitNextToCloseFruit")
    go_back_to_scan_point = State("GoBackToScanPoint")
    calculate_next_safe_point = State("CalculateNextSafePoint")
    navigate_to_safe_point = State("NavigateToSafePoint")

    # Final run states
    check_for_remaining_targets = State("CheckForRemainingTargets")
    go_to_target_fruit = State("GoToTargetFruit")
    sit_next_to_target_fruit = State("SitNextToTargetFruit")

    end_state = State("EndState", final=True)

    ## Transitions
    T_scan_to_approach_fruit = scan.to(approach_fruit)
    # Approach fruit loop
    T_approach_fruit_to_sit_next_to_close_fruit = approach_fruit.to(sit_next_to_close_fruit)
    T_sit_next_to_close_fruit_to_go_back_to_scan_point = sit_next_to_close_fruit.to(go_back_to_scan_point)
    T_go_back_to_scan_point_to_approach_fruit = go_back_to_scan_point.to(approach_fruit)

    T_scan_to_calculate_next_safe_point = scan.to(calculate_next_safe_point)
    T_go_back_to_scan_point_to_calculate_next_safe_point = go_back_to_scan_point.to(calculate_next_safe_point)
    T_calculate_next_safe_point_to_navigate_to_safe_point = calculate_next_safe_point.to(navigate_to_safe_point)

    T_navigate_to_safe_point_to_scan = navigate_to_safe_point.to(scan)

    # Final run states transitions
    T_calculate_next_safe_point_to_check_for_remaining_targets = calculate_next_safe_point.to(check_for_remaining_targets)
    T_check_for_remaining_targets_to_go_to_target_fruit = check_for_remaining_targets.to(go_to_target_fruit)
    T_go_to_target_fruit_to_sit_next_to_target_fruit = go_to_target_fruit.to(sit_next_to_target_fruit)
    T_sit_next_to_target_fruit_to_go_to_target_fruit = sit_next_to_target_fruit.to(go_to_target_fruit)
    
    # Termination transitions
    T_go_back_to_scan_point_to_end_state = go_back_to_scan_point.to(end_state)
    T_check_for_remaining_targets_to_end_state = check_for_remaining_targets.to(end_state)
    T_sit_next_to_target_fruit_to_end_state = sit_next_to_target_fruit.to(end_state)

class PiBotFruitSearchSMLevel3(StateMachine):
    ## ALL STATES
    # Mapping states
    scan = State("Scan", initial=True)
    go_to_target = State("GoToTarget")
    wait_at_target = State("WaitAtTarget")

    end_state = State("EndState", final=True)

    ## Transitions
    T_scan_to_go_to_target = scan.to(go_to_target)
    T_go_to_target_to_wait_at_target = go_to_target.to(wait_at_target)
    T_go_to_target_to_scan = go_to_target.to(scan)
    T_wait_at_target_to_scan = wait_at_target.to(scan)
    T_go_to_target_to_end_state = go_to_target.to(end_state)



'''
robotSM = PiBotFruitSearchSM()

graph = DotGraphMachine(robotSM)
dot = graph()
dot.write_png("robot_machine.png")
for _ in range(4):
    print(f"State: {robot.current_state.id}")
    if robot.current_state == robot.searching:
        if not robot.search_to_fruit():
            robot.search_to_navigate()
    elif robot.current_state == robot.approach_fruit:
        robot.fruit_to_navigate()
    elif robot.current_state == robot.navigate:
        robot.navigate_to_search()
'''