#!/bin/python3

from colorama import Fore, Back, Style, init


def showAllColors():
    init(autoreset=True)

    names = [
        "RED", "LIGHTRED_EX",
        "YELLOW", "LIGHTYELLOW_EX",
        "GREEN", "LIGHTGREEN_EX",
        "CYAN", "LIGHTCYAN_EX",
        "BLUE", "LIGHTBLUE_EX",
        "MAGENTA", "LIGHTMAGENTA_EX",
        "BLACK", "LIGHTBLACK_EX",
        "WHITE", "LIGHTWHITE_EX"
    ]

    colors = {
        "Fore": [
            Fore.RED, Fore.LIGHTRED_EX,
            Fore.YELLOW, Fore.LIGHTYELLOW_EX,
            Fore.GREEN, Fore.LIGHTGREEN_EX,
            Fore.CYAN, Fore.LIGHTCYAN_EX,
            Fore.BLUE, Fore.LIGHTBLUE_EX,
            Fore.MAGENTA, Fore.LIGHTMAGENTA_EX,
            Fore.BLACK, Fore.LIGHTBLACK_EX,
            Fore.WHITE, Fore.LIGHTWHITE_EX
        ],
        "Back": [
            Back.RED, Back.LIGHTRED_EX,
            Back.YELLOW, Back.LIGHTYELLOW_EX,
            Back.GREEN, Back.LIGHTGREEN_EX,
            Back.CYAN, Back.LIGHTCYAN_EX,
            Back.BLUE, Back.LIGHTBLUE_EX,
            Back.MAGENTA, Back.LIGHTMAGENTA_EX,
            Back.BLACK, Back.LIGHTBLACK_EX,
            Back.WHITE, Back.LIGHTWHITE_EX
        ]
    }
    print(Style.BRIGHT + "======== BACKGROUND COLORS ========")
    for idx, (color, name) in enumerate(zip(colors["Back"], names)):
        print(f'{idx}: {color} Back.{name} {Style.RESET_ALL}')

    print(Style.BRIGHT + "\n======== FOREGROUND COLORS ========")
    for idx, (color, name) in enumerate(zip(colors["Fore"], names)):
        print(f'{idx}: {color} Fore.{name}')

    print(Style.RESET_ALL)
    print(Style.BRIGHT + "============== STYLE ==============")
    print(f'Bold: {Style.BRIGHT}Style.BRIGHT{Style.RESET_ALL}')


if __name__ == "__main__":
    showAllColors()