import re
import sys

from sage.interfaces.magma import MagmaGBLogPrettyPrinter

from ..Experiment import Logger, nolog


class MagmaGBLogger(MagmaGBLogPrettyPrinter):
    def __init__(self, verbosity=1, style="magma", log: Logger = nolog):
        self._log_degs = log.sublogger("degs")
        self._log_pairs = log.sublogger("pairs")
        self._degs = []
        self._pairs = []
        super().__init__(verbosity, style)

    def get_stats(self):
        """Return tuple of degrees and number of critical pairs encountered during computation.
        If magma crashed, the last value are the current ones when crashed.
        """
        return (self._degs, self._pairs)

    def write(self, s):
        """
        EXAMPLES::

            sage: P.<x,y,z> = GF(32003)[]
            sage: I = sage.rings.ideal.Katsura(P)
            sage: _ = I.groebner_basis('magma',prot=True) # indirect doctest, optional - magma
            ********************
            FAUGERE F4 ALGORITHM
            ********************
            ...
            Total Faugere F4 time: ..., real time: ...
        """
        verbosity, style = self.verbosity, self.style

        if isinstance(s, bytes):
            # sys.stdout.encoding can be None or something else
            if isinstance(sys.stdout.encoding, str):
                s = s.decode(sys.stdout.encoding)
            else:
                s = s.decode("UTF-8")

        if self.storage:
            s = self.storage + s
            self.storage = ""

        for line in s.splitlines():
            # deal with the Sage <-> Magma syncing code
            match = re.match(MagmaGBLogPrettyPrinter.cmd_inpt, line)
            if match:
                self.sync = 1
                continue

            if self.sync:
                if self.sync == 1:
                    self.sync = line
                    continue
                else:
                    if line == "":
                        continue
                    self.sync = None
                    continue

            if re.match(MagmaGBLogPrettyPrinter.app_inpt, line):
                continue

            if re.match(MagmaGBLogPrettyPrinter.deg_curr, line):
                match = re.match(MagmaGBLogPrettyPrinter.deg_curr, line)

                nbasis, npairs, deg, npairs_deg = map(int, match.groups())

                self.curr_deg = deg
                self.curr_npairs = npairs
                self._degs.append(deg)
                self._pairs.append(npairs)

                self._log_degs.log(self.curr_deg, show=False)
                self._log_pairs.log(self.curr_npairs, show=False)

            if re.match(MagmaGBLogPrettyPrinter.pol_curr, line):
                match = re.match(MagmaGBLogPrettyPrinter.pol_curr, line)
                pol_curr, col_curr = map(int, match.groups())

                if pol_curr != 0:
                    if self.max_deg < self.curr_deg:
                        self.max_deg = self.curr_deg

                    if style == "sage" and verbosity >= 1:
                        print(
                            "Leading term degree: %2d. Critical pairs: %d."
                            % (self.curr_deg, self.curr_npairs)
                        )
                else:
                    if style == "sage" and verbosity >= 1:
                        print(
                            "Leading term degree: %2d. Critical pairs: %d (all pairs of current degree eliminated by criteria)."
                            % (self.curr_deg, self.curr_npairs)
                        )

            if style == "magma" and verbosity >= 1:
                print(line)
