from datetime import datetime
from io import TextIOWrapper


def get_logger(f: TextIOWrapper = None, to_print: bool = False):

    def _f(s: str, print_this_time: bool = False):
        timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M%S')}]"
        s = f'{timestamp} {s}'
        if f:
            f.write(s + '\n')
        if to_print:
            print(s)
        elif print_this_time:
            print(s)

    return _f
