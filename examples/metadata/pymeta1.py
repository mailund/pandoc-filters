#!/usr/bin/env python3

import sys
from pandocfilters import toJSONFilter, Meta

def separator(key, value, format, meta):
    if key == "Meta":
        print(value)
        print(meta)
        return None

    else:
        # do not change anything.
        return None

if __name__ == '__main__':
    toJSONFilter(separator)
