from statemachine import StateMachine, State
from statemachine.contrib.diagram import DotGraphMachine

class PiBotFruitSearchSM(StateMachine):
    ## ALL STATES
    # Mapping states
    spin = State("Spin", initial=True)
    approach_fruit = State("ApproachFruit")
    sit_next_to_close_fruit = State("SitNextToCloseFruit")
    go_back_to_spin_point = State("GoBackToSpinPoint")
    calculate_next_safe_point = State("CalculateNextSafePoint")
    navigate_to_safe_point = State("NavigateToSafePoint")

    # Final run states
    check_for_remaining_targets = State("CheckForRemainingTargets")
    go_to_target_fruit = State("GoToTargetFruit")
    sit_next_to_target_fruit = State("SitNextToTargetFruit")

    end_state = State("EndState", final=True)

    ## Transitions
    T_spin_to_approach_fruit = spin.to(approach_fruit)
    # Approach fruit loop
    T_approach_fruit_to_sit_next_to_close_fruit = approach_fruit.to(sit_next_to_close_fruit)
    T_sit_next_to_close_fruit_to_go_back_to_spin_point = sit_next_to_close_fruit.to(go_back_to_spin_point)
    T_go_back_to_spin_point_to_approach_fruit = go_back_to_spin_point.to(approach_fruit)

    T_spin_to_calculate_next_safe_point = spin.to(calculate_next_safe_point)
    T_go_back_to_spin_point_to_calculate_next_safe_point = go_back_to_spin_point.to(calculate_next_safe_point)
    T_calculate_next_safe_point_to_navigate_to_safe_point = calculate_next_safe_point.to(navigate_to_safe_point)

    T_navigate_to_safe_point_to_spin = navigate_to_safe_point.to(spin)

    # Final run states transitions
    T_calculate_next_safe_point_to_check_for_remaining_targets = calculate_next_safe_point.to(check_for_remaining_targets)
    T_check_for_remaining_targets_to_go_to_target_fruit = check_for_remaining_targets.to(go_to_target_fruit)
    T_go_to_target_fruit_to_sit_next_to_target_fruit = go_to_target_fruit.to(sit_next_to_target_fruit)
    T_sit_next_to_target_fruit_to_go_to_target_fruit = sit_next_to_target_fruit.to(go_to_target_fruit)
    
    # Termination transitions
    T_go_back_to_spin_point_to_end_state = go_back_to_spin_point.to(end_state)
    T_check_for_remaining_targets_to_end_state = check_for_remaining_targets.to(end_state)
    T_sit_next_to_target_fruit_to_end_state = sit_next_to_target_fruit.to(end_state)

robotSM = PiBotFruitSearchSM()

graph = DotGraphMachine(robotSM)
dot = graph()
dot.write_png("robot_machine.png")


'''
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