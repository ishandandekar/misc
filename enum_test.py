from enum import Enum


class State(Enum):
    PLAYING = 1
    PAUSED = 2
    EXIT = 0


state = State.EXIT

if state.value == 0:
    print("Yeah...")

if state == State.EXIT:
    print("I need guns...")

if State.PAUSED.value > State.PLAYING.value:
    print("Lots of guns...")
