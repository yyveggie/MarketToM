#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Expert Perspective Library and Related Utility Functions
Used for multi-expert approach in market action probability calculation
"""

import random
from typing import List

PERSPECTIVE_LIBRARY = [
    "A contrarian strategist who interprets strong consensus between intention and emotion as a peak signal for a market reversal.",
    "A momentum and herding analyst who views powerful emotion as the fuel that will amplify and sustain the market's intention.",
    "An intention-emotion mismatch detector who specializes in finding dissonance between what the market wants to do and how it feels.",
    "A Prospect Theory risk analyst who assesses the reliability of the market's intention by analyzing the nature of its emotion, such as loss aversion or the 'house money' effect.",
    "A narrative strength assessor who treats market intention as the 'plot' and emotion as the 'tone' to evaluate the coherence of the market's story.",
    "An 'animal spirits' barometer who assesses whether the market's intention is driven by calculated reasoning or by raw, non-rational emotional tides.",
    "A regret aversion specialist who analyzes how the collective desire to avoid future regret will amplify or suppress the market's stated intention.",
    "A second-level thinker who predicts market moves by anticipating how sophisticated players will react to the obvious collective intention and emotion of the herd.",
    "An emotional volatility analyst who believes a highly unstable emotional state makes the stated market intention unreliable and prone to sudden reversals.",
    "A cognitive dissonance specialist who predicts the market's next action by anticipating the resolution of psychological tension between its intention and emotion."
]

def get_random_perspectives(num_perspectives: int) -> List[str]:
    """
    Randomly select a specified number of expert roles from the perspective library
    
    Args:
        num_perspectives: Number of expert perspectives needed
        
    Returns:
        List of randomly selected expert perspective descriptions
    """
    num_to_select = min(num_perspectives, len(PERSPECTIVE_LIBRARY))
    
    selected_perspectives = random.sample(PERSPECTIVE_LIBRARY, num_to_select)
    
    if num_perspectives > len(PERSPECTIVE_LIBRARY):
        additional_needed = num_perspectives - len(PERSPECTIVE_LIBRARY)
        additional_perspectives = [random.choice(PERSPECTIVE_LIBRARY) for _ in range(additional_needed)]
        selected_perspectives.extend(additional_perspectives)
    
    return selected_perspectives

def extend_perspective_library(new_perspectives: List[str]) -> None:
    """
    Add new expert perspective descriptions to the library
    
    Args:
        new_perspectives: List of expert perspective descriptions to add
    """
    global PERSPECTIVE_LIBRARY
    for perspective in new_perspectives:
        if perspective not in PERSPECTIVE_LIBRARY:
            PERSPECTIVE_LIBRARY.append(perspective)

def get_all_perspectives() -> List[str]:
    """
    Get all available expert perspective descriptions
    
    Returns:
        List of all available expert perspective descriptions
    """
    return PERSPECTIVE_LIBRARY.copy() 