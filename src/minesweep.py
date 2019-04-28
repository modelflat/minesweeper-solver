import logging
from collections import defaultdict

import numpy
import pyautogui

logging.basicConfig(level="INFO", format="%(asctime)s @%(funcName)s [%(levelname)s] -- %(message)s")

BOX_SIZE = 16
SMILEYFACE_SIZE = 26
ID_PIXEL = (9, 12)

COL2NUM = {
    (0, 0, 255): 1,
    (0, 128, 0): 2,
    (255, 0, 0): 3,
    (0, 0, 128): 4,
    (128, 0, 0): 5,
    (0, 128, 128): 6,
    (0, 0, 0): 7,
    (128, 128, 128): 8
}

pyautogui.PAUSE = 0.02


def find_minesweeper_field(shape):
    loc = pyautogui.locateOnScreen("../img/smiley_face.png", grayscale=True)
    if loc is None:
        raise RuntimeError("Cannot find minesweeper smiley face")
    x, y = loc.left, loc.top
    # go to center of smiley face
    x += 13
    y += 13
    reset_loc = (x, y)

    # go to field's top left corner
    y += 14 + 13
    x -= BOX_SIZE * shape[0] // 2
    field_loc = (x, y)

    return SweepSweep(shape, field_loc, reset_loc)


def number_of(field_img, x, y):
    x, y = x * BOX_SIZE, y * BOX_SIZE + 40

    # is flag?
    col = field_img.getpixel((x + 5, y + 5))
    if col[0] == 255:
        return 9

    # is empty?
    col = field_img.getpixel((x + 1, y + 1))
    if col[0] == 255 and col[1] == 255 and col[2] == 255:
        return 0

    # is number / which number?
    col = field_img.getpixel((x  + ID_PIXEL[0], y + ID_PIXEL[1]))
    num = COL2NUM.get(col)
    if num is None:
        col = field_img.getpixel((x + 2, y + 2))
        if col[0] == 255:
            return 0
        return -1
    elif num == 7:
        col = field_img.getpixel((x + 1, y + 1))
        if col[0] == 255 and col[1] == 0:
            return 0xdead
        return 7
    else:
        return num


class SweepSweep:

    def __init__(self, shape, field_loc, reset_loc):
        self.fs = shape
        self.floc = field_loc
        self.rloc = reset_loc
        self.death_pixel = (reset_loc[0] - 4, reset_loc[1] + 4)

        # region for taking screenshot
        self.game_region = (
            self.floc[0],
            self.floc[1] - 40,
            BOX_SIZE*self.fs[0],
            BOX_SIZE*self.fs[1] + 40,
        )
        self.smiley_face_loc_x = BOX_SIZE * self.fs[0] // 2 - 13

        # state:
        self.field = numpy.zeros(shape, dtype=numpy.int32)
        self.is_dead = False
        self.has_won = False

    def fl2sc(self, x, y):
        return self.floc[0] + x * BOX_SIZE + 1, self.floc[1] + y * BOX_SIZE + 1

    def update_state(self):
        field_img = pyautogui.screenshot(region=self.game_region)

        self.is_dead = field_img.getpixel((self.smiley_face_loc_x + 9, 17))[0] == 0
        self.has_won = field_img.getpixel((self.smiley_face_loc_x + 6, 13))[0] == 0

        # update field
        for x in range(self.fs[0]):
            for y in range(self.fs[1]):
                self.field[x, y] = number_of(field_img, x, y)

    def show_field(self, **kwargs):

        def symbol(x, **kwargs):
            if x == 0xdead:
                return "X"
            if x == 0:
                return kwargs.get("empty", ".")
            if x == -1:
                return kwargs.get("open", "_")
            if x == 9:
                return kwargs.get("flag", "&")
            return str(x)
        s = "\n".join([" ".join(map(lambda x: symbol(x, **kwargs), line)) for line in self.field.T])
        print(s)

    def reset(self):
        logging.info("Resetting state...")
        pyautogui.click(*self.rloc, _pause=False)
        self.update_state()

    def dig(self, x, y, update=True):
        pyautogui.click(*self.fl2sc(x, y), _pause=True)
        if update:
            self.update_state()

    def dig_random(self):
        pyautogui.click(*self.fl2sc(numpy.random.randint(0, self.fs[0]), numpy.random.randint(0, self.fs[1])))#, _pause=False)
        self.update_state()

    def mark(self, x, y):
        self.field[x, y] = 9
        pyautogui.click(*self.fl2sc(x, y), button="right")#, _pause=False)

    def mark_batch(self, cells, update=True, twice=False):
        for cell in set(cells):
            self.mark(*cell)
            if twice:
                self.mark(*cell)
        if update:
            self.update_state()

    def dig_batch(self, cells, update=True):
        for cell in set(cells):
            self.dig(*cell, update=False)
        if update:
            self.update_state()

    def for_each(self, f):
        for x in range(self.fs[0]):
            for y in range(self.fs[1]):
                f(self.field[x, y], x, y)

    def for_each_number(self, f):
        for x in range(self.fs[0]):
            for y in range(self.fs[1]):
                el = self.field[x, y]
                if el <= 0 or el > 8: continue
                f(el, x, y)

    def for_each_neighbour_cell(self, x, y, f):
        for xx in range(max(0, x - 1), min(self.fs[0] - 1, x + 1) + 1):
            for yy in range(max(0, y - 1), min(self.fs[1] - 1, y + 1) + 1):
                if xx == x and yy == y: continue
                f(self.field[xx, yy], xx, yy)

    def count_empty(self):
        cnt = 0

        def _f(n, x, y):
            nonlocal cnt
            if n == -1:
                cnt += 1

        self.for_each(_f)
        return cnt

    def start(self, strategy="random", **kwargs):
        logging.info("Starting a game using '{}' strategy".format(strategy))
        sw.update_state()
        if strategy == "random":
            n = kwargs.get("empty_estimate", 16)
            n_attempts = kwargs.get("tries", 10)
            i = n_attempts
            while self.count_empty() < n and i > 0:
                self.dig_random()
                if self.is_dead:
                    self.reset()
                    i = n_attempts
                self.update_state()
                i -= 1
        else:
            raise RuntimeError("No such strategy: {}".format(strategy))

    def reduce_field(self):
        def _f(n, x, y):
            def _ff(nn, xx, yy):
                if nn == 9:
                    self.field[x, y] -= 1
            self.for_each_neighbour_cell(x, y, _ff)
            if self.field[x, y] == 0 or self.field[x, y] < -1:
                self.field[x, y] = -1
        self.for_each_number(_f)

    def eliminate_trivial(self, dbg=False):
        cells_to_mark = []
        cells_to_dig = []

        def _f(n, x, y):
            nonlocal cells_to_mark
            empty_neighbours = []
            marked_neighbours = 0

            def _fn(nn, xx, yy):
                nonlocal empty_neighbours, marked_neighbours
                if nn == 0:
                    empty_neighbours.append((xx, yy))
                elif nn == 9:
                    marked_neighbours += 1
            self.for_each_neighbour_cell(x, y, _fn)

            if marked_neighbours > n:
                raise RuntimeError("marked_neighbours > n")

            if n == marked_neighbours:
                cells_to_dig.extend(empty_neighbours)

            if len(empty_neighbours) <= n - marked_neighbours:
                cells_to_mark.extend(empty_neighbours)

        self.for_each_number(_f)
        self.mark_batch(cells_to_mark, update=False)
        if dbg:
            self.mark_batch(cells_to_dig, twice=True)
        else:
            self.dig_batch(cells_to_dig)

        success = bool(cells_to_dig) or bool(cells_to_mark)
        if success:
            logging.info("Simple calculation succeeded -- {} can be dug, {} can be marked".format(
                len(cells_to_dig), len(cells_to_mark)
            ))
        else:
            logging.info("Simple calculation cannot find any cells to dig or mark")
        return success

    def eliminate_constrained(self, dbg=False):
        constraints = defaultdict(lambda: set())

        self.reduce_field()

        def _f(n, x, y):
            def _gather_empty(nn, xx, yy):
                nonlocal constraints
                if nn == 0:
                    constraints[(n, x, y)].add((xx, yy))
            self.for_each_neighbour_cell(x, y, _gather_empty)
        self.for_each_number(_f)

        cells_to_dig = set()
        cells_to_mark = set()

        for cell, cell_constr in constraints.items():
            n, x, y = cell
            for another_cell, another_cell_constr in constraints.items():
                nn, xx, yy = another_cell
                common = cell_constr.intersection(another_cell_constr)

                if n == nn and (common == cell_constr or common == another_cell_constr):
                    cells_to_dig.update(cell_constr.symmetric_difference(another_cell_constr))

                if n > nn:
                    # can mark some cells of bigger area
                    belongs_to_bigger_only = cell_constr.difference(another_cell_constr)
                    if len(belongs_to_bigger_only) == n - nn:
                        cells_to_mark.update(belongs_to_bigger_only)

        if dbg:
            self.mark_batch(cells_to_mark)
            self.mark_batch(cells_to_dig, twice=True)
        else:
            self.mark_batch(cells_to_mark, update=False)
            self.dig_batch(cells_to_dig)

        success = bool(cells_to_dig) or bool(cells_to_mark)
        if success:
            logging.info("Constraint analysis succeeded -- {} can be dug, {} can be marked".format(
                len(cells_to_dig), len(cells_to_mark)
            ))
        else:
            logging.info("Constraint analysis cannot find any cells to dig or mark")
        return success

    def dig_least_dangerous_at_random(self):
        cells = defaultdict(lambda: 0)

        self.reduce_field()

        def _f(n, x, y):
            empty = 0

            def _count_empty(nn, xx, yy):
                nonlocal empty
                if nn == 0:
                    empty += 1
            self.for_each_neighbour_cell(x, y, _count_empty)

            prob = 1.0 / empty

            def _ff(nn, xx, yy):
                if nn == 0:
                    cells[(xx, yy)] = max(cells[(xx, yy)], prob)

            self.for_each_neighbour_cell(x, y, _ff)
        self.for_each_number(_f)

        if not cells:
            # should not happen most of the time, but can under normal conditions
            # e.g. when all remaining fields are isolated from numbers
            return None

        least_dangerous = sorted(cells.items(), key=lambda x: x[1])[0]
        return least_dangerous

    def solve_trivial(self, starting_strategy="random"):
        self.start(strategy=starting_strategy)

        any_changes = True
        while any_changes and not self.has_won:
            # 1.
            while self.eliminate_trivial(): pass
            # 2.
            any_changes = self.eliminate_constrained()
            # 3.
            if not any_changes and not self.has_won:
                # TODO think of other techniques?
                logging.info("Further analysis seems complicated. Let's pick one at random")

                least_dangerous = self.dig_least_dangerous_at_random()
                if least_dangerous is None:
                    raise NotImplementedError("NOT IMPLEMENTED -- isolated mines")
                else:
                    logging.info(least_dangerous)
                    self.dig(*least_dangerous[0])
                any_changes = not self.is_dead

        if self.has_won:
            logging.info("CONGRATS U WON")
        elif self.is_dead:
            logging.info("LOL U DIED")

        return self.has_won

    def solve_or_die(self, n_attempts=3):
        for attempt in range(n_attempts):
            self.reset()
            logging.info("Solving (attempt {}/{})".format(attempt, n_attempts))
            if self.solve_trivial():
                break


if __name__ == "__main__":
    sw = find_minesweeper_field((30, 16))
    sw.solve_or_die()
