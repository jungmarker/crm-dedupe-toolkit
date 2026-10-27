import pandas as pd
import pytest

from crmdedupe import accounts as A, normalize as N
from crmdedupe.cluster import clusters
from crmdedupe.model import LinkageModel


@pytest.mark.parametrize("raw,expected", [("The Brown-Ramirez Co.", "brown ramirez"), ("CHAVEZ INC LLC", "chavez"),
                                          ("Lewis Inc Inc", "lewis"), ("Inc", "inc")])
def test_company_name(raw, expected):
    assert N.company_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [("https://www.Acme.io/about", "acme.io"), ("acme.io", "acme.io"),
                                          (None, ""), ("www.acme.io", "acme.io")])
def test_domain(raw, expected):
    assert N.domain(raw) == expected


def test_state_phone_email_names():
    assert N.state("Maryland") == N.state("md") == N.state("MD") == "MD"
    assert N.phone("(301) 875-7182") == N.phone("+1 301 875 7182") == N.phone("301.875.7182") == "3018757182"
    assert N.phone("12345") == ""
    assert N.email(" JSmith@Acme.COM ") == "jsmith@acme.com"
    assert N.email("jsmithacme.com") == N.email("jsmith@") == ""
    assert N.first_name("Bob") == N.first_name("robert") == "robert"


def test_clusters_are_transitive():
    c = clusters(["a", "b", "c", "d"], [("a", "b"), ("c", "b")])
    assert c["a"] == c["b"] == c["c"] == "a" and c["d"] == "d"


def test_name_levels_separate_identical_from_shared_root():
    acc = pd.DataFrame({"account_id": ["1", "2", "3"], "account_name": ["Rich PLC Holdings", "RICH PLC HOLDINGS", "Rich LLC"],
                        "website": ["www.richplc.com", "richplc.com", "www.richllc.com"], "billing_state": ["IL", "Illinois", "TX"],
                        "industry": ["Retail"] * 3, "employee_count": [15, 15, 900], "owner_id": ["u1", "u2", "u3"]})
    a = A.prepare(acc)
    lv = A.compare(a, pd.DataFrame({"id_l": ["1", "1"], "id_r": ["2", "3"]}))
    assert lv.name.tolist() == [4, 3]
    assert lv.domain.tolist() == [2, 1] and lv.state.tolist() == [2, 1] and lv.employees.tolist() == [3, 1]


def test_model_save_load_roundtrip(tmp_path):
    lv = pd.DataFrame({c.name: [len(c.levels) - 1] * 5 + [1] * 5 for c in A.COMPARISONS})
    m = LinkageModel(A.COMPARISONS).fit(lv, [1] * 5 + [0] * 5)
    m.save(tmp_path / "m.json")
    m2 = LinkageModel.load(tmp_path / "m.json", A.COMPARISONS)
    assert (abs(m.predict(lv) - m2.predict(lv)) < 1e-12).all()
    assert m.predict(lv)[0] > 0.5 > m.predict(lv)[-1]
