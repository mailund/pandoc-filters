#!/usr/bin/env python3

import sys


from panflute import *

no_exercise = 1
id_map = {}

def number_exercises(elem, doc):
    global no_exercise
    if type(elem) == Div and "Exercise" in elem.classes:

        meta = doc.get_metadata()

        if doc.format == "latex":
            exercise_env = "exercises"
            if "exercise_env" in meta:
                exercise_env = meta["exercise_env"]

            if elem.identifier:
                label = r"\label{" + elem.identifier + "}"
            else:
                label = ""

            block = [
                RawBlock(r"\begin{" + exercise_env + "}" + label,
                         "tex"),
                elem,
                RawBlock(r"\end{" + exercise_env + "}",
                         "tex")
            ]
            return block

    if type(elem) == Span:
        if not "out" in elem.classes:
            return elem
        if doc.format not in elem.classes:
            return []
        else:
            return elem.content.list


    if type(elem) == Cite:
        actual_cite = elem.citations[0]
        identifier = actual_cite.id
        if not identifier.startswith("ex:"):
            return elem

        if doc.format == "html":
            return [
                Link(*actual_cite.prefix.list, url = "#" + identifier)
            ]
        if doc.format == "latex":
            return actual_cite.prefix.list + [
                RawInline(r"\ref{" + identifier + "}", "latex")
            ]

run_filter(number_exercises)
