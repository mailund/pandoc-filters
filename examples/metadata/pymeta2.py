#!/usr/bin/env python3

import sys
from string import Template  # using .format() is hard because of {} in tex
import panflute as pf

TEMPLATE_GLS = Template(r"\gls{$acronym}")
TEMPLATE_NEWACRONYM = Template(r"\newacronym{$acronym}{$acronym}{$definition}")


def prepare(doc):
    print(doc.get_metadata(), file=sys.stderr)
    print(doc.format, file=sys.stderr)


def action(e, doc):
    pass


def main(doc=None):
    return pf.run_filter(prepare=prepare, action=action, doc=doc)


if __name__ == '__main__':
    main()
