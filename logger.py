from io import TextIOWrapper


def get_logger(f: TextIOWrapper = None, to_print: bool = False):

    def _f(s: str, print_this_time: bool = False):
        if f:
            f.write(s + '\n')
        if to_print:
            print(s)
        elif print_this_time:
            print(s)

    return _f
