import re
import random
import ast
import operator
import os


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    reward_mode = os.getenv('COUNTDOWN_REWARD_FUNCTION', 'level1')
    
    if reward_mode == 'level1':
        return level1_reward(solution_str, ground_truth, method, format_score, score)
    elif reward_mode == 'level2':
        return level2_reward(solution_str, ground_truth, method, format_score, score)
    elif reward_mode == 'level3':
        return level3_reward(solution_str, ground_truth, method, format_score, score)
    elif reward_mode == 'level4':
        return level4_reward(solution_str, ground_truth, method, format_score, score)
    else:
        raise ValueError(f"Unknown reward mode: {reward_mode}")

def level1_reward(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """Level 1: Basic binary reward function that returns:
    - 0.0 if no valid equation is found
    - format_score (0.1) if equation is found but incorrect
    - full score (1.0) if equation is correct
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score

def level2_reward(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """Level 2: Enhanced reward function that provides:
    - 0.0 if no valid equation is found
    - format_score (0.1) for parseable equation
    - validation_reward (0.5) for correct number usage
    - full score (1.0) for correct solution
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    do_print = random.randint(1, 64) == 1
    equation = extract_solution(solution_str=solution_str)
    
    if do_print:
        print(f"--------------------------------")
        print("Reward mode: format_validation_reward")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Give format reward for having parseable equation
    if do_print:
        print(f"Equation parsed, giving format score: {format_score}")
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
    
    # Give enhanced reward for valid number usage
    validation_reward = min(5 * format_score, score)
    if do_print:
        print(f"Valid numbers used, giving validation reward: {validation_reward}")
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return validation_reward
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return validation_reward
    except:
        if do_print:
            print(f"Error evaluating equation")
        return validation_reward

def level3_reward(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """Level 3: Reward function with partial credit that provides:
    - 0.0 if no valid equation is found
    - Partial reward based on ratio of correct numbers used between 0.1 and 0.5
    - validation_reward (0.5) for correct number usage
    - full score (1.0) for correct solution
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print("Reward mode: partial_numbers_reward")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Calculate overlap ratio and base validation reward
    numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
    overlap = len(set(numbers_in_eq).intersection(set(numbers)))
    overlap_ratio = overlap / len(numbers)
    validation_reward = min(5 * format_score, score)
    
    # Determine partial reward based on number overlap
    if not validate_equation(equation, numbers):
        partial_reward = format_score + (validation_reward - format_score) * overlap_ratio
        partial_reward = min(score, partial_reward)
        if do_print:
            print(f"Number overlap ratio: {overlap_ratio}")
            print(f"Partial reward based on number overlap: {partial_reward}")    
        return partial_reward
    
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return validation_reward
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return validation_reward
    except:
        if do_print:
            print(f"Error evaluating equation")
        return validation_reward

def level4_reward(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """Level 4: Full-featured reward function that provides:
    - 0.0 if no valid equation is found
    - Partial reward based on ratio of correct numbers used between 0.1 and 0.5
    - validation_reward (0.5) for correct number usage
    - evaluation_reward (0.7) for valid arithmetic even if wrong answer
    - full score (1.0) for correct solution
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print("Reward mode: comprehensive_reward")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Calculate overlap ratio and base validation reward
    numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
    overlap = len(set(numbers_in_eq).intersection(set(numbers)))
    overlap_ratio = overlap / len(numbers)
    validation_reward = min(5 * format_score, score)
    
    # Determine partial reward based on number overlap
    if not validate_equation(equation, numbers):
        partial_reward = format_score + (validation_reward - format_score) * overlap_ratio
        partial_reward = min(score, partial_reward)
        if do_print:
            print(f"Number overlap ratio: {overlap_ratio}")
            print(f"Partial reward based on number overlap: {partial_reward}")    
        return partial_reward
    
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return validation_reward
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            # Enhanced reward for successful evaluation even with wrong answer
            eval_reward = min(7 * format_score, score)
            if do_print:
                print(f"Wrong result but valid evaluation: equation = {result}, target = {target}")
                print(f"Giving enhanced evaluation reward: {eval_reward}")
            return eval_reward
    except:
        if do_print:
            print(f"Error evaluating equation")
        return validation_reward
