"""Import CSV into a spectrum: reading the two file formats, working out the
spectrum, and the Tcl scripts SpecTcl is sent. SpecTcl is played by a small
model that understands exactly the scripts the importer writes."""

import fnmatch
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

from services import csv_import as ci


# ------------------------------------------------------------ InterSpec CSV

INTERSPEC_HEAD = """Spectrum Name:
Description:
Detector Used:
Comments:
Acquisition Mode:,PHA Pre-Amp
High Voltage:,0
Conversion Gain:,1024
Fine Gain:,1.00
Calibration Coefficients:,-751.0,33.0
Calibration Channels:,179,189
Calibration Energies:,5156.0,5486.0
Elapsed Live Time:,7200.0
Start Time:,August 18, 2025, 04:59:18
ROIs:
ROI:,Low,High,Gross,Net,FWHM,Centroid


Channel Data:
Channel,Energy,Counts
"""


def interspec_file(tmp_path, counts=(0, 1, 13, 63, 2), name="spec.csv", head=INTERSPEC_HEAD):
    rows = "".join("%d,%.1f,%d\n" % (ch, -751.0 + 33.0 * ch, n)
                   for ch, n in enumerate(counts))
    path = tmp_path / name
    path.write_text(head + rows)
    return str(path)


def plain_file(tmp_path, text, name="plain.csv"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_interspec_csv_gives_counts_against_energy(tmp_path):
    x, y = ci.read_interspec_csv(interspec_file(tmp_path))
    assert np.allclose(x, [-751.0, -718.0, -685.0, -652.0, -619.0])
    assert np.allclose(y, [0, 1, 13, 63, 2])


def test_interspec_header_numbers_are_not_taken_for_data(tmp_path):
    """The header has rows of numbers of its own (calibration channels and
    energies, an ROI column row); only the rows under the column names count."""
    x, y = ci.read_interspec_csv(interspec_file(tmp_path, counts=(7, 8)))
    assert len(x) == 2 and np.allclose(y, [7, 8])


def test_interspec_csv_without_energy_uses_the_channel(tmp_path):
    path = plain_file(tmp_path, "Comments:\nChannel,Counts\n0,4\n1,5\n2,6\n")
    x, y = ci.read_interspec_csv(path)
    assert np.allclose(x, [0, 1, 2]) and np.allclose(y, [4, 5, 6])


def test_interspec_csv_with_energy_and_data_columns(tmp_path):
    path = plain_file(tmp_path, "Energy, Data\r\n0.5, 3\r\n1.5, 9\r\n")
    x, y = ci.read_interspec_csv(path)
    assert np.allclose(x, [0.5, 1.5]) and np.allclose(y, [3, 9])


def test_interspec_table_ends_at_the_first_row_that_is_not_data(tmp_path):
    path = interspec_file(tmp_path, counts=(1, 2, 3))
    with open(path, "a") as f:
        f.write("\nSomething else:\n9,9,9\n")
    x, y = ci.read_interspec_csv(path)
    assert np.allclose(y, [1, 2, 3])


def test_a_file_without_a_spectrum_table_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ci.read_interspec_csv(plain_file(tmp_path, "0,0\n1,1\n2,4\n"))
    with pytest.raises(ValueError):                     # column names, no rows
        ci.read_interspec_csv(plain_file(tmp_path, INTERSPEC_HEAD, "empty.csv"))


# ------------------------------------------------------------ either format

def test_read_csv_takes_plain_rows_as_spectcl(tmp_path):
    x, y = ci.read_csv(plain_file(tmp_path, "0,0\n1,1\n2,4\n3,9\n"), "SpecTcl")
    assert np.allclose(x, [0, 1, 2, 3]) and np.allclose(y, [0, 1, 4, 9])


def test_read_csv_takes_one_column_as_counts_against_the_row_number(tmp_path):
    x, y = ci.read_csv(plain_file(tmp_path, "5\n6\n7\n"), "SpecTcl")
    assert np.allclose(x, [0, 1, 2]) and np.allclose(y, [5, 6, 7])


def test_read_csv_sorts_by_x_and_drops_rows_that_are_not_numbers(tmp_path):
    x, y = ci.read_csv(plain_file(tmp_path, "x,y\n2,20\n0,5\n1,nan\n3,30\n"), "SpecTcl")
    assert np.allclose(x, [0, 2, 3]) and np.allclose(y, [5, 20, 30])


def test_read_csv_reads_interspec_when_asked(tmp_path):
    x, y = ci.read_csv(interspec_file(tmp_path), "InterSpec")
    assert len(x) == 5 and y[3] == 63


def test_read_csv_refuses_a_file_of_the_other_program(tmp_path):
    with pytest.raises(ValueError):                     # numpy: ragged rows
        ci.read_csv(interspec_file(tmp_path), "SpecTcl")
    with pytest.raises(ValueError):
        ci.read_csv(plain_file(tmp_path, "0,0\n1,1\n"), "InterSpec")


def test_read_csv_refuses_a_file_without_numbers(tmp_path):
    with pytest.raises(ValueError):
        ci.read_csv(plain_file(tmp_path, "a,b\nc,d\n"), "SpecTcl")


# ------------------------------------------------------------ name and axis

def test_the_default_name_is_the_file_name_without_its_extension():
    assert ci.default_spectrum_name("/data/run12.csv") == "run12"
    assert ci.default_spectrum_name("Th228.source.txt") == "Th228.source"
    assert ci.default_spectrum_name("noext") == "noext"


def test_the_default_name_replaces_what_a_spectrum_name_cannot_have():
    name = ci.default_spectrum_name("/data/Run 12 [Th228] {a}*?.csv")
    assert name == "Run_12__Th228___a___"
    assert ci.spectrum_name_problem(name) == ""
    assert ci.default_spectrum_name(".csv") != ""


@pytest.mark.parametrize("name", ["run12", "a.b.c", "run_12-(Th228)+x:y", "α-spectrum"])
def test_usable_names(name):
    assert ci.spectrum_name_problem(name) == ""


@pytest.mark.parametrize("name", ["", "two words", "a{b", "a}b", "a[1]", "a*", "a?",
                                  "a\\b", 'a"b', "a$b", "a;b", "tab\there"])
def test_names_that_are_refused(name):
    assert ci.spectrum_name_problem(name) != ""


def test_the_axis_puts_each_x_value_at_the_centre_of_its_bin():
    low, high, bins, even = ci.axis_for(-751.0 + 33.0 * np.arange(1024))
    assert (low, high, bins, even) == (-767.5, 33024.5, 1024, True)
    width = (high - low) / bins
    assert width == 33.0
    assert low + 0.5 * width == -751.0                  # first bin centre


def test_the_axis_tolerates_rounded_x_values():
    x = np.round(0.3333333 * np.arange(500), 2)         # two decimals in the file
    assert ci.axis_for(x)[3] is True


def test_the_axis_reports_x_values_that_are_not_evenly_spaced():
    x = 2.0 * np.arange(200) + 0.01 * np.arange(200) ** 2    # quadratic calibration
    low, high, bins, even = ci.axis_for(x)
    assert even is False and bins == 200
    assert low < x[0] and high > x[-1]


def test_the_axis_needs_two_distinct_rows():
    with pytest.raises(ValueError):
        ci.axis_for([5.0])
    with pytest.raises(ValueError):
        ci.axis_for([1.0, 2.0, 2.0, 3.0])


def test_counts_become_whole_numbers_and_never_negative():
    assert ci.channel_counts([0, 1.4, 1.6, -3, 2.5e3]) == [0, 1, 2, 0, 2500]
    assert all(type(v) is int for v in ci.channel_counts(np.array([1.0, 2.0])))


# ------------------------------------------------------------ SpecTcl

class FakeSpecTcl:
    """Stands in for PyREST against a SpecTcl that knows `spectrum`,
    `parameter`, `treeparameter`, `channel` and `sbind`, as far as the
    importer uses them."""

    def __init__(self):
        self.parameters = {}
        self.spectra = {}       # name -> dict(parameter, low, high, bins, counts)
        self.bound = []
        self.scripts = []
        self.fail = None        # substring of the script that is to fail
        self.dead = False

    def script(self, command):
        self.scripts.append(command)
        if self.dead:
            return None
        if self.fail and self.fail in command:
            return {"status": "ERROR", "detail": "it broke"}
        try:
            return {"status": "OK", "detail": self._run(command)}
        except Exception as e:
            return {"status": "ERROR", "detail": str(e)}

    def _run(self, command):
        m = re.fullmatch(r"if \{\[llength \[parameter -list \{(\S+)\}\]\] == 0\} "
                         r"\{treeparameter -create \{(\S+)\} (\S+) (\S+) (\d+) \{\}\}", command)
        if m:
            assert m.group(1) == m.group(2)
            if m.group(1) not in self.parameters:
                self.parameters[m.group(1)] = (float(m.group(3)), float(m.group(4)), int(m.group(5)))
            return ""
        m = re.fullmatch(r"parameter -list \{(\S+)\}", command)
        if m:
            names = sorted(n for n in self.parameters if fnmatch.fnmatchcase(n, m.group(1)))
            return " ".join("{%s %d {}}" % (n, i) for i, n in enumerate(names))
        m = re.fullmatch(r"spectrum -list \{(\S+)\}", command)
        if m:
            s = self.spectra.get(m.group(1))
            if s is None:
                return ""
            return "{7 %s 1 %s {{%g %g %d}} long}" % (
                m.group(1), s["parameter"], s["low"], s["high"], s["bins"])
        m = re.fullmatch(r"spectrum -delete \{(\S+)\}", command)
        if m:
            del self.spectra[m.group(1)]
            if m.group(1) in self.bound:
                self.bound.remove(m.group(1))
            return ""
        m = re.fullmatch(r"spectrum \{(\S+)\} 1 \{(\S+)\} \{\{(\S+) (\S+) (\d+)\}\}", command)
        if m:
            name, parameter = m.group(1), m.group(2)
            if name in self.spectra:
                raise ValueError("Duplicate spectrum " + name)
            if parameter not in self.parameters:
                raise ValueError("No such parameter " + parameter)
            bins = int(m.group(5))
            self.spectra[name] = dict(parameter=parameter, low=float(m.group(3)),
                                      high=float(m.group(4)), bins=bins,
                                      counts=[0] * bins)
            return name
        m = re.fullmatch(r"namespace eval (\S+) \{variable v; variable channel (\d+); "
                         r"foreach v \{([\d ]+)\} "
                         r"\{channel -set \{(\S+)\} \$channel \$v; incr channel\}\}; "
                         r"namespace delete (\S+)", command)
        if m:
            assert m.group(1) == m.group(5) == ci.TCL_NAMESPACE
            counts = self.spectra[m.group(4)]["counts"]
            channel = int(m.group(2))
            for v in m.group(3).split(" "):
                counts[channel] = int(v)                # IndexError past the end
                channel += 1
            return ""
        m = re.fullmatch(r"sbind \{(\S+)\}", command)
        if m:
            assert m.group(1) in self.spectra
            self.bound.append(m.group(1))
            return ""
        raise ValueError("not a script the importer writes: " + command)


def filled_by(scripts, name, bins):
    """The channel contents after running fill scripts on an empty spectrum."""
    tcl = FakeSpecTcl()
    tcl.parameters["p"] = (0, 1, 1)
    tcl.script("spectrum {%s} 1 {p} {{0 1 %d}}" % (name, bins))
    for script in scripts:
        assert tcl.script(script)["status"] == "OK", script
    return tcl.spectra[name]["counts"]


def test_fill_scripts_set_every_channel():
    rng = np.random.default_rng(1)
    counts = [int(v) for v in rng.integers(0, 100000, 3000)]
    scripts = ci.fill_scripts("s", counts)
    assert len(scripts) > 1
    assert filled_by(scripts, "s", 3000) == counts


def test_fill_scripts_stay_short_enough_for_a_url():
    counts = [4000000000] * 5000
    scripts = ci.fill_scripts("a.long.spectrum.name", counts, max_length=500)
    assert max(map(len, scripts)) <= 500
    assert filled_by(scripts, "a.long.spectrum.name", 5000) == counts


def test_fill_scripts_leave_out_the_empty_stretches_at_the_ends():
    counts = [0] * 2000 + [5, 0, 7] + [0] * 2000
    scripts = ci.fill_scripts("s", counts)
    assert len(scripts) == 1
    assert "variable channel 2000; foreach v {5 0 7} " in scripts[0]
    assert filled_by(scripts, "s", len(counts)) == counts


def test_fill_scripts_for_a_sparse_spectrum():
    rng = np.random.default_rng(2)
    counts = [int(v) if rng.random() < 0.02 else 0 for v in rng.integers(1, 50, 20000)]
    scripts = ci.fill_scripts("s", counts, max_length=400)
    assert filled_by(scripts, "s", len(counts)) == counts


def test_an_empty_spectrum_needs_no_fill_script():
    assert ci.fill_scripts("s", [0] * 100) == []


def test_fill_scripts_keep_their_variables_out_of_the_global_namespace():
    """The REST script service runs at global level; a bare loop there would
    overwrite variables of the user's. Both loop variables are declared in a
    namespace of the importer's, which is deleted again."""
    for script in ci.fill_scripts("s", [1, 2, 3]):
        assert script.startswith("namespace eval %s {variable v; variable channel 0; "
                                 % ci.TCL_NAMESPACE)
        assert script.endswith("; namespace delete " + ci.TCL_NAMESPACE)


def test_no_script_uses_a_command_spectcl_redefines():
    """SpecTcl has commands of its own under names Tcl also uses. `apply`
    there applies a gate to a spectrum; a fill script written with Tcl's
    `apply` failed on the real SpecTcl with "Failed search of dictionary"."""
    tcl = FakeSpecTcl()
    ci.import_spectrum(tcl, "a", [0.0, 1.0, 2.0], [1, 2, 3], replace=False)
    ci.import_spectrum(tcl, "a", [0.0, 1.0, 2.0], [1, 2, 3], replace=True)
    ci.spectrum_exists(tcl, "a")
    spectcl_only = {"apply", "gate", "clear", "fold", "project", "filter",
                    "pseudo", "attach", "start", "stop", "integrate", "fit",
                    "version", "tape", "unbind", "ungate"}
    for script in tcl.scripts:
        words = set(re.findall(r"[A-Za-z_:]+", script))
        assert not (words & spectcl_only), script


def test_import_creates_fills_and_binds_in_that_order():
    tcl = FakeSpecTcl()
    x = -751.0 + 33.0 * np.arange(6)
    low, high, bins, parameter = ci.import_spectrum(tcl, "run12", x, [0, 1, 13, 63, 2, 0])

    assert (low, high, bins, parameter) == (-767.5, -569.5, 6, "import.001")
    spectrum = tcl.spectra["run12"]
    assert spectrum["counts"] == [0, 1, 13, 63, 2, 0]
    assert (spectrum["low"], spectrum["high"], spectrum["bins"]) == (-767.5, -569.5, 6)
    assert spectrum["parameter"] == "import.001" and "import.001" in tcl.parameters
    assert tcl.bound == ["run12"]
    kinds = [s.split(" ")[0] for s in tcl.scripts]
    assert kinds == ["parameter", "if", "spectrum", "namespace", "sbind"]


def test_each_imported_spectrum_gets_a_parameter_of_its_own():
    """Gates are defined on parameters: spectra sharing one share their gates."""
    tcl = FakeSpecTcl()
    for name in ("a", "b", "c"):
        ci.import_spectrum(tcl, name, [0.0, 1.0, 2.0], [1, 2, 3])
    assert [tcl.spectra[n]["parameter"] for n in ("a", "b", "c")] == [
        "import.001", "import.002", "import.003"]
    assert sorted(tcl.parameters) == ["import.001", "import.002", "import.003"]


def test_the_numbering_goes_on_after_the_highest_parameter_there_is():
    tcl = FakeSpecTcl()
    for existing in ("import.002", "import.017", "import.x", "important.900",
                     "my.import.500", "import.03.b"):
        tcl.parameters[existing] = (0, 1, 1)
    assert ci.next_parameter(tcl) == "import.018"
    tcl.parameters["import.999"] = (0, 1, 1)
    assert ci.next_parameter(tcl) == "import.1000"


def test_replacing_an_imported_spectrum_keeps_its_parameter():
    """So the gates made on it are still on it after the file is imported again."""
    tcl = FakeSpecTcl()
    ci.import_spectrum(tcl, "a", [0.0, 1.0, 2.0], [1, 2, 3])
    ci.import_spectrum(tcl, "b", [0.0, 1.0, 2.0], [1, 2, 3])
    *_, parameter = ci.import_spectrum(tcl, "a", [0.0, 1.0], [9, 8], replace=True)
    assert parameter == "import.001" and tcl.spectra["a"]["parameter"] == "import.001"
    assert sorted(tcl.parameters) == ["import.001", "import.002"]


def test_replacing_a_spectrum_that_was_not_imported_makes_a_new_parameter():
    tcl = FakeSpecTcl()
    tcl.parameters["adc.00"] = (0, 1, 1)
    tcl.script("spectrum {a} 1 {adc.00} {{0 4096 4096}}")
    assert ci.parameter_of(tcl, "a") == "" and ci.parameter_of(tcl, "none") == ""
    *_, parameter = ci.import_spectrum(tcl, "a", [0.0, 1.0], [9, 8], replace=True)
    assert parameter == "import.001" and "adc.00" in tcl.parameters


@pytest.mark.parametrize("definition, expected", [
    ("{7 run12 1 import.004 {{0 10 10}} long}", "import.004"),
    ("{7 run12 1 {import.004} {{0 10 10}} long}", "import.004"),
    ("{7 run12 1 {import.004 } {{0 10 10}} long}", "import.004"),
    ("{7 run12 1 { import.004 } {{0 10 10}} long}", "import.004"),
    ("{7 {run12} 1 {import.004} {{0 10 10}} long}", "import.004"),
    ("{7 run12 1 adc.00 {{0 10 10}} long}", ""),
    ("{7 other 1 import.004 {{0 10 10}} long}", ""),
    ("", ""),
])
def test_the_parameter_is_read_off_the_spectrum_definition(definition, expected):
    rest = type("R", (), {"script": lambda self, c: {"status": "OK", "detail": definition}})()
    assert ci.parameter_of(rest, "run12") == expected


def test_an_existing_spectrum_is_found_and_replaced_only_on_request():
    tcl = FakeSpecTcl()
    assert ci.spectrum_exists(tcl, "a") is False
    ci.import_spectrum(tcl, "a", [0.0, 1.0, 2.0], [1, 2, 3])
    assert ci.spectrum_exists(tcl, "a") is True

    with pytest.raises(ci.CsvImportError) as err:
        ci.import_spectrum(tcl, "a", [0.0, 1.0], [9, 9])
    assert "Duplicate spectrum a" in str(err.value)
    assert tcl.spectra["a"]["counts"] == [1, 2, 3]      # the old one is untouched

    ci.import_spectrum(tcl, "a", [0.0, 1.0], [9, 8], replace=True)
    assert tcl.spectra["a"]["counts"] == [9, 8] and tcl.bound == ["a"]


@pytest.mark.parametrize("failing", ["channel -set", "sbind"])
def test_a_failure_after_the_spectrum_was_made_removes_it_again(failing):
    tcl = FakeSpecTcl()
    tcl.fail = failing
    with pytest.raises(ci.CsvImportError) as err:
        ci.import_spectrum(tcl, "a", [0.0, 1.0, 2.0], [1, 2, 3])
    assert "it broke" in str(err.value)
    assert "a" not in tcl.spectra and tcl.bound == []


def test_a_failure_to_create_says_what_spectcl_said():
    tcl = FakeSpecTcl()
    tcl.fail = "spectrum {a}"
    with pytest.raises(ci.CsvImportError) as err:
        ci.import_spectrum(tcl, "a", [0.0, 1.0], [1, 2])
    assert "creating the spectrum" in str(err.value) and "it broke" in str(err.value)


def test_no_answer_from_spectcl_is_an_error_not_a_crash():
    tcl = FakeSpecTcl()
    tcl.dead = True
    with pytest.raises(ci.CsvImportError) as err:
        ci.spectrum_exists(tcl, "a")
    assert "No usable answer" in str(err.value)
    with pytest.raises(ci.CsvImportError):
        ci.import_spectrum(tcl, "a", [0.0, 1.0], [1, 2])


def test_the_interspec_sample_end_to_end(tmp_path):
    counts = [0] * 150 + [3, 40, 220, 61, 5] + [0] * 869
    path = interspec_file(tmp_path, counts=counts, name="Th228 run.csv")
    x, y = ci.read_csv(path, "InterSpec")
    name = ci.default_spectrum_name(path)
    tcl = FakeSpecTcl()
    ci.import_spectrum(tcl, name, x, y)
    spectrum = tcl.spectra["Th228_run"]
    assert spectrum["counts"] == counts and spectrum["bins"] == 1024
    # the tallest channel sits at the energy the file gives for it
    width = (spectrum["high"] - spectrum["low"]) / spectrum["bins"]
    assert spectrum["low"] + (152 + 0.5) * width == pytest.approx(-751.0 + 33.0 * 152)


# ------------------------------------------------------------ the REST call

def test_pyrest_script_sends_the_script_and_returns_the_reply():
    import logging
    from PyREST import PyREST
    rest = PyREST(logging.getLogger("t"), "host", "8080")
    sent = []

    def answer(url):
        sent.append(url)
        return b'{"status": "OK","detail": "3"}'

    rest.sendRequest = answer
    assert rest.script("spectrum -list {a b}") == {"status": "OK", "detail": "3"}
    assert sent == ["http://host:8080/spectcl/script?command=spectrum+-list+%7Ba+b%7D"]


@pytest.mark.parametrize("answer", [None, b"<html>404 Not Found</html>", b"[1, 2]"])
def test_pyrest_script_returns_none_for_anything_that_is_not_a_reply(answer):
    import logging
    from PyREST import PyREST
    rest = PyREST(logging.getLogger("t"), "host", "8080")
    rest.sendRequest = lambda url: answer
    assert rest.script("expr 1") is None
