"""Normalization: turn messy CRM strings into comparable keys."""
import re

# Legal suffixes and filler words that don't identify a company
COMPANY_STOPWORDS = {"the", "inc", "incorporated", "llc", "l", "c", "ltd", "limited", "corp", "corporation",
                     "co", "company", "plc", "group", "holdings", "and"}

STATES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
          "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
          "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
          "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
          "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
          "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
          "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
          "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
          "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
          "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"}

# Common English nicknames → canonical first name
NICKNAMES = {"bob": "robert", "rob": "robert", "bill": "william", "will": "william", "liz": "elizabeth",
             "beth": "elizabeth", "jen": "jennifer", "jenny": "jennifer", "mike": "michael", "chris": "christopher",
             "kate": "katherine", "katie": "katherine", "jim": "james", "jimmy": "james", "rick": "richard",
             "dick": "richard", "dan": "daniel", "danny": "daniel", "matt": "matthew", "joe": "joseph",
             "tom": "thomas", "tony": "anthony", "pat": "patricia", "maggie": "margaret", "meg": "megan",
             "jess": "jessica", "sam": "samantha", "nick": "nicholas", "andy": "andrew", "steve": "steven",
             "dave": "david", "greg": "gregory", "jeff": "jeffrey", "ken": "kenneth", "ron": "ronald",
             "tim": "timothy", "sue": "susan", "becky": "rebecca", "alex": "alexander", "ben": "benjamin"}


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return str(s).strip()


def company_name(name) -> str:
    """'The Brown-Ramirez Co.' → 'brown ramirez'."""
    tokens = re.sub(r"[^a-z0-9]+", " ", _clean(name).lower()).split()
    core = [t for t in tokens if t not in COMPANY_STOPWORDS]
    return " ".join(core or tokens)


def domain(url) -> str:
    """'https://www.Acme.io/about' → 'acme.io'."""
    d = re.sub(r"^[a-z]+://", "", _clean(url).lower())
    d = d.split("/")[0].split("?")[0]
    return d[4:] if d.startswith("www.") else d


def state(s) -> str:
    """'Maryland' / 'md' / 'MD' → 'MD'."""
    s = _clean(s)
    return STATES.get(s.lower(), s.upper())


def email(e) -> str:
    """Lower-case and trim; return '' for addresses that can't be valid."""
    e = _clean(e).lower()
    return e if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", e) else ""


def email_domain(e) -> str:
    e = email(e)
    return e.split("@")[1] if e else ""


FREE_MAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com", "proton.me"}


def phone(p) -> str:
    """Keep the last 10 digits: '(301) 875-7182' / '+1 301 875 7182' → '3018757182'."""
    digits = re.sub(r"\D", "", _clean(p))
    return digits[-10:] if len(digits) >= 10 else ""


def first_name(n) -> str:
    n = _clean(n).lower()
    return NICKNAMES.get(n, n)


def person_name(n) -> str:
    return re.sub(r"[^a-z]", "", _clean(n).lower())
