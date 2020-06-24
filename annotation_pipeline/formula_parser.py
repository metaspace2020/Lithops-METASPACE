import re
from collections import Counter

clean_regexp = re.compile(r'[.=]')
formula_regexp = re.compile(r'([A-Z][a-z]*)([0-9]*)')
adduct_split_regexp = re.compile(r'([+-]*)([A-Za-z0-9]+)')


class ParseFormulaError(Exception):
    def __init__(self, message):
        self.message = message


def parse_formula(f):
    return [(elem, int(n or '1'))
            for (elem, n) in formula_regexp.findall(f)]


def format_modifiers(*adducts):
    return ''.join(adduct for adduct in adducts if adduct and adduct not in ('[M]+', '[M]-'))


def generate_ion_formula(formula, *adducts):
    formula = clean_regexp.sub('', formula)
    adducts = [clean_regexp.sub('', adduct) for adduct in adducts]

    ion_elements = Counter(dict(parse_formula(formula)))

    for adduct in adducts:
        op, a_formula = adduct_split_regexp.findall(adduct)[0]
        assert op in ('+','-'), 'Adduct should be prefixed with + or -'
        if op == '+':
            for elem, n in parse_formula(a_formula):
                ion_elements[elem] += n
        else:
            for elem, n in parse_formula(a_formula):
                ion_elements[elem] -= n
                if ion_elements[elem] < 0:
                    raise ParseFormulaError(f'Negative total element count for {elem}')

    # Ordered per https://en.wikipedia.org/wiki/Chemical_formula#Hill_system
    if ion_elements['C'] != 0:
        element_order = ['C', 'H', *sorted(key for key in ion_elements.keys() if key not in ('C', 'H'))]
    elif any(count > 0 for count in ion_elements.values()):
        element_order = sorted(key for key in ion_elements.keys())
    else:
        raise ParseFormulaError('No remaining elements')

    ion_formula_parts = []
    for elem in element_order:
        count = ion_elements[elem]
        if count != 0:
            ion_formula_parts.append(elem)
            if count > 1:
                ion_formula_parts.append(str(count))

    return ''.join(ion_formula_parts)


def safe_generate_ion_formula(*parts):
    try:
        return generate_ion_formula(*(part for part in parts if part))
    except ParseFormulaError:
        return None
