
# VERSION <version-name> PROBLEMTYPE <problem-type> DISCOUNTFACTOR <discount-factor>
# OBSERVATIONS INTS ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* DOUBLES
# ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* CHARCOUNT <char-count> ACTIONS INTS
# ([times-to-repeat-this-tuple=1] <min-value> <max-value>)* DOUBLES ([times-to-repeat-this-tuple=1]
# <min-value> <max-value>)* CHARCOUNT <char-count> REWARDS (<min-value> <max-value>) EXTRA
# [extra text of your choice goes here]";


class TaskSpec:
    def __init__(self, discount_factor=1.0, reward_range=(-1, 1)):
        self.version = "RL-Glue-3.0"
        self.prob_type = "episodic"
        self._discount_factor = discount_factor
        self._act = {}
        self._obs = {}
        self._act_charcount = 0
        self._obs_charcount = 0
        self._reward_range = reward_range
        self._extras = ""

    def to_taskspec(self):
        ts_list = ["VERSION " + self.version,
                   "PROBLEMTYPE " + self.prob_type,
                   "DISCOUNTFACTOR " + str(self._discount_factor)]

        # Observations
        if len(self._obs.keys()) > 0:
            ts_list += ["OBSERVATIONS"]
            if "INTS" in self._obs:
                ts_list += ["INTS"] + self._obs["INTS"]
            if "DOUBLES" in self._obs:
                ts_list += ["DOUBLES"] + self._obs["DOUBLES"]
            if "CHARCOUNT" in self._obs:
                ts_list += ["CHARCOUNT"] + self._obs["CHARCOUNT"]

        # Actions
        if len(self._act.keys()) > 0:
            ts_list += ["ACTIONS"]
            if "INTS" in self._act:
                ts_list += ["INTS"] + self._act["INTS"]
            if "DOUBLES" in self._act:
                ts_list += ["DOUBLES"] + self._act["DOUBLES"]
            if "CHARCOUNT" in self._act:
                ts_list += ["CHARCOUNT"] + self._act["CHARCOUNT"]

        ts_list += ["REWARDS", "(" + str(self._reward_range[0]) + " " + str(self._reward_range[1]) + ")"]
        if self._extras != "":
            ts_list += ["EXTRAS", self._extras]
        return ' '.join(ts_list)

    def set_discount_factor(self, factor):
        self._discount_factor = factor

    def set_continuing(self):
        self.prob_type = "continuing"

    def set_episodic(self):
        self.prob_type = "episodic"

    def set_problem_type_custom(self, prob_type):
        self.prob_type = prob_type

    def add_act(self, range_, repeat=1, type_="INTS"):
        rept = "" if repeat <= 1 else str(repeat) + " "
        self._act.setdefault(type_, []).append("(" + rept + str(range_[0]) + " " + str(range_[1]) + ")")

    def add_obs(self, range_, repeat=1, type_="INTS"):
        rept = "" if repeat <= 1 else str(repeat) + " "
        self._obs.setdefault(type_, []).append("(" + rept + str(range_[0]) + " " + str(range_[1]) + ")")

    def add_int_act(self, range_, repeat=1):
        self.add_act(map(int, range_), repeat, "INTS")

    def add_int_obs(self, range_, repeat=1):
        self.add_obs(map(int, range_), repeat, "INTS")

    def add_double_act(self, range_, repeat=1):
        self.add_act(range_, repeat, "DOUBLES")

    def add_double_obs(self, range_, repeat=1):
        self.add_obs(range_, repeat, "DOUBLES")

    def set_charcount_act(self, charcount):
        self._act["CHARCOUNT"] = [str(charcount)]

    def set_charcount_obs(self, charcount):
        self._obs["CHARCOUNT"] = [str(charcount)]

    def set_reward_range(self, low, high):
        self._reward_range = (low, high)

    def set_extra(self, extra):
        self._extras = extra
