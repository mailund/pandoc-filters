#!/usr/bin/env python3

import sys


from panflute import *

no_exercise = 1

def number_exercises(elem, doc):
    global no_exercise
    if type(elem) == Div and "Exercise" in elem.classes:

        meta = doc.get_metadata()

        if doc.format == "latex":
            exercise_env = "exercises"
            if "exercise_env" in meta:
                exercise_env = meta["exercise_env"]
            block = [
                RawBlock(r"\begin{" + exercise_env + "}", "tex"),
                elem,
                RawBlock(r"\end{" + exercise_env + "}", "tex")
            ]
            return block

        if doc.format == "html":
            level = 1
            if "exercise_header_level" in meta:
                level = int(meta["exercise_header_level"])

            title = [Str("Exercise"), Space, Str(str(no_exercise))]
            no_exercise += 1
            return [Header(*title, level = level, classes = elem.classes), elem]

        return elem

run_filter(number_exercises)
