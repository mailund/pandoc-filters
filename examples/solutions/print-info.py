#!/usr/bin/env python3

import sys
from panflute import *

def print_structure(elem, doc):
    if type(elem) == Div:
        print("id:", elem.identifier, file = sys.stderr)
        print("classes:", elem.classes, file = sys.stderr)
        print("attributes:", elem.attributes, file = sys.stderr)

run_filter(print_structure)
