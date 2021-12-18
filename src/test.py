from prepare import prepare_data


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    pass


def print_success(func_name: str):
    print(f'{func_name}: {bcolors.OKGREEN}SUCCESS!{bcolors.ENDC}')


def print_fail(func_name: str, reason: str = None):
    print(f'{func_name} {bcolors.FAIL}FAIL{bcolors.ENDC}' +
          f': {reason}' if reason is not None else '')


if __name__ == '__main__':
    main()
